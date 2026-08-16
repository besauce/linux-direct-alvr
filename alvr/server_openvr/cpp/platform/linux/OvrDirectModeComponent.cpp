#define _GNU_SOURCE
#include <link.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include "OvrDirectModeComponent.h"

#include "alvr_server/Logger.h"
#include "alvr_server/Settings.h"
#include "alvr_server/include/openvr_math.h"
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include <cstring>
#include <linux/dma-buf.h>
#include <linux/sync_file.h>
#include <poll.h>
#include <sys/ioctl.h>

namespace {

constexpr uint64_t kFenceStatWindowNs = 1'000'000'000;

enum class FenceKind { None, Stub, Real, RealActive };

// Ask the kernel WHO the current write fence on a dma-buf belongs to.
// EXPORT_SYNC_FILE never returns nothing: with no matching fence on the
// reservation object it substitutes an already-signalled stub fence
// (dma-buf.c, dma_fence_get_stub), whose driver name is the literal string
// "stub" (dma-fence.c). A real fence reports the amdgpu scheduler's names.
// This is the discriminator between "no fence was ever attached" and "a
// fence is attached and it was simply already signalled when we looked".
FenceKind queryFenceKind(
    int dmabufFd, char* driverNameOut, size_t driverNameLen, uint64_t* signalNsOut = nullptr
) {
#if defined(DMA_BUF_IOCTL_EXPORT_SYNC_FILE) && defined(SYNC_IOC_FILE_INFO)
    struct dma_buf_export_sync_file exportInfo = {};
    exportInfo.flags = DMA_BUF_SYNC_READ;
    exportInfo.fd = -1;
    if (ioctl(dmabufFd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &exportInfo) != 0) {
        return FenceKind::None;
    }

    FenceKind kind = FenceKind::None;
    struct sync_file_info info = {};
    if (ioctl(exportInfo.fd, SYNC_IOC_FILE_INFO, &info) == 0 && info.num_fences > 0) {
        std::vector<struct sync_fence_info> fences(info.num_fences);
        info.sync_fence_info = (uint64_t)(uintptr_t)fences.data();
        if (ioctl(exportInfo.fd, SYNC_IOC_FILE_INFO, &info) == 0) {
            bool anyActive = false;
            bool anyStub = false;
            uint64_t maxSignalNs = 0;
            for (uint32_t i = 0; i < info.num_fences; ++i) {
                if (strcmp(fences[i].driver_name, "stub") == 0) {
                    anyStub = true;
                } else if (fences[i].status == 0) {
                    // A real fence that has NOT signaled yet: the writer's
                    // GPU work is in flight right now.
                    anyActive = true;
                } else if (fences[i].timestamp_ns > maxSignalNs) {
                    maxSignalNs = fences[i].timestamp_ns;
                }
            }
            if (anyActive) {
                kind = FenceKind::RealActive;
            } else if (maxSignalNs > 0) {
                kind = FenceKind::Real;
            } else if (anyStub) {
                kind = FenceKind::Stub;
            }
            if (signalNsOut != nullptr) {
                *signalNsOut = maxSignalNs;
            }
            if (driverNameOut != nullptr && driverNameLen > 0) {
                snprintf(
                    driverNameOut,
                    driverNameLen,
                    "%s/%s",
                    fences[0].driver_name,
                    fences[0].obj_name
                );
            }
        }
    }
    close(exportInfo.fd);

    return kind;
#else
    (void)dmabufFd;
    (void)driverNameOut;
    (void)driverNameLen;
    (void)signalNsOut;
    return FenceKind::None;
#endif
}

}

OvrDirectModeComponent::OvrDirectModeComponent(std::shared_ptr<PoseHistory> poseHistory)
    : m_poseHistory(poseHistory)
    , m_submitLayer(0) {
    const char* rewarp = getenv("ALVR_REWARP");
    if (rewarp != nullptr && strcmp(rewarp, "0") == 0) {
        m_rewarp = false;
    }
    const char* deadband = getenv("ALVR_REWARP_DEADBAND_URAD");
    if (deadband != nullptr) {
        m_rewarpDeadbandUrad = atof(deadband);
    }
    const char* presentSleep = getenv("ALVR_PRESENT_SLEEP_MS");
    if (presentSleep != nullptr) {
        m_presentSleepMs = atoi(presentSleep);
        Info("OvrDirectMode: present write-race sleep %d ms (ALVR_PRESENT_SLEEP_MS)\n",
             m_presentSleepMs);
    }
    const char* gpuWait = getenv("ALVR_GPU_WAIT");
    if (gpuWait != nullptr && strcmp(gpuWait, "0") == 0) {
        m_gpuWait = false;
    }
    Info(
        "OvrDirectMode: GPU-side write fence wait %s (ALVR_GPU_WAIT)\n",
        m_gpuWait ? "enabled" : "disabled"
    );
    const char* runningStart = getenv("ALVR_RUNNING_START_US");
    if (runningStart != nullptr) {
        m_runningStartUs = atoi(runningStart);
    }
    const char* noVsyncEvents = getenv("ALVR_NO_VSYNC_EVENTS");
    if (noVsyncEvents != nullptr && strcmp(noVsyncEvents, "0") != 0) {
        m_sendVsyncEvents = false;
    }
    Info(
        "OvrDirectMode: running start %d us, vsync events %s\n",
        m_runningStartUs,
        m_sendVsyncEvents ? "on" : "off (hardcoded-offset model)"
    );
    if (m_rewarp) {
        Info(
            "OvrDirectMode: rotational reprojection on (ALVR_REWARP=0 disables), deadband %.0f urad\n",
            m_rewarpDeadbandUrad
        );
    }

    m_encodeWorker = std::thread([this] { EncodeWorkerLoop(); });
}

OvrDirectModeComponent::~OvrDirectModeComponent() {
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
        m_workerExit = true;
        if (m_jobPending) {
            for (int e = 0; e < 2; ++e) {
                if (m_job.waitFds[e] >= 0) {
                    close(m_job.waitFds[e]);
                }
            }
            m_jobPending = false;
        }
    }
    m_jobCv.notify_one();
    if (m_encodeWorker.joinable()) {
        m_encodeWorker.join();
    }

    // These dups are only ever replaced in place on a texture set switch, so
    // the last set is still open here and this is the only place it is closed.
    for (int& fd : m_syncFds) {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }
}

void OvrDirectModeComponent::DrainPendingJob() {
    std::lock_guard<std::mutex> lock(m_jobMutex);
    if (m_jobPending) {
        for (int e = 0; e < 2; ++e) {
            if (m_job.waitFds[e] >= 0) {
                close(m_job.waitFds[e]);
            }
        }
        m_jobPending = false;
        m_jobsDropped++;
    }
}

void OvrDirectModeComponent::SetViewParams(const FfiViewParams params[2]) {
    std::unique_lock<std::mutex> lock(m_viewParamsMutex);
    m_viewParams[0] = params[0];
    m_viewParams[1] = params[1];
    m_viewParamsValid = true;
}

// Export the sync file a reader must wait on (the writer's fences) from a
// dma-buf fd. Returns the sync_file fd, or -1 when there is nothing to wait
// on or the kernel lacks the ioctl. The caller owns the returned fd.
static int exportReadSyncFile(int dmabufFd) {
#if defined(DMA_BUF_IOCTL_EXPORT_SYNC_FILE)
    struct dma_buf_export_sync_file exportInfo = {};
    exportInfo.flags = DMA_BUF_SYNC_READ;
    exportInfo.fd = -1;
    if (ioctl(dmabufFd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &exportInfo) != 0) {
        return -1;
    }
    return exportInfo.fd;
#else
    (void)dmabufFd;
    return -1;
#endif
}

// A zero or badly denormalized quaternion is an invalid pose (the Rust side
// substitutes an all-zero motion when tracking has no head sample). Engaging
// the warp on one would push an identity warp while moving the stamp, and the
// moved stamp may name a timestamp the head pose queue never saw, which drops
// the frame on the Rust side. The capture path carried the same guard as its
// zero-pose detection.
static bool IsUnitQuat(double w, double x, double y, double z) {
    double normSq = w * w + x * x + y * y + z * z;
    return normSq > 0.9 && normSq < 1.1;
}

// Fill the warp push constants for a rotational reprojection from
// renderOrientation to latestOrientation. Backward sampling rotation:
// source-from-destination view rotation R_render^T * R_latest; with the
// Hamilton product convention of openvr_math.h, conj(q_render) * q_latest
// composes the same way. Both quats are world-from-head, so the world frame
// cancels and the playspace transform never enters. Returns false (params
// left disabled) on degenerate FOV, i.e. before view params arrive, and on
// invalid orientations.
static bool BuildWarpParams(
    const vr::HmdQuaternion_t& renderOrientation,
    const FfiQuat& latestOrientation,
    const FfiFov& leftFov,
    const FfiFov& rightFov,
    alvr::render::WarpParams& params
) {
    float leftTans[4]
        = { tanf(leftFov.left), tanf(leftFov.right), tanf(leftFov.up), tanf(leftFov.down) };
    float rightTans[4]
        = { tanf(rightFov.left), tanf(rightFov.right), tanf(rightFov.up), tanf(rightFov.down) };
    if (leftTans[1] - leftTans[0] == 0.f || rightTans[1] - rightTans[0] == 0.f
        || leftTans[2] - leftTans[3] == 0.f || rightTans[2] - rightTans[3] == 0.f) {
        return false;
    }
    if (!IsUnitQuat(
            renderOrientation.w, renderOrientation.x, renderOrientation.y, renderOrientation.z
        )
        || !IsUnitQuat(
            latestOrientation.w, latestOrientation.x, latestOrientation.y, latestOrientation.z
        )) {
        return false;
    }

    vr::HmdQuaternion_t qLatest
        = { latestOrientation.w, latestOrientation.x, latestOrientation.y, latestOrientation.z };
    vr::HmdQuaternion_t qDelta = vrmath::quaternionConjugate(renderOrientation) * qLatest;

    vr::HmdMatrix34_t rot;
    HmdMatrix_QuatToMat(qDelta.w, qDelta.x, qDelta.y, qDelta.z, &rot);

    // GLSL mat4 is column-major: rotation[col * 4 + row]
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            params.rotation[col * 4 + row]
                = (row < 3 && col < 3) ? rot.m[row][col] : (float)(row == col);
        }
    }
    memcpy(params.leftTans, leftTans, sizeof(leftTans));
    memcpy(params.rightTans, rightTans, sizeof(rightTans));
    params.enabled = 1;
    return true;
}

void OvrDirectModeComponent::EncodeWorkerLoop() {
    while (true) {
        FrameJob job;
        {
            std::unique_lock<std::mutex> lock(m_jobMutex);
            m_jobCv.wait(lock, [this] { return m_jobPending || m_workerExit; });
            if (m_workerExit) {
                // Belt and braces: a Present racing teardown could have
                // repopulated the slot after the destructor's drain.
                if (m_jobPending) {
                    for (int e = 0; e < 2; ++e) {
                        if (m_job.waitFds[e] >= 0) {
                            close(m_job.waitFds[e]);
                        }
                    }
                    m_jobPending = false;
                }
                break;
            }
            job = m_job;
            m_jobPending = false;
        }

        auto const workStart = std::chrono::steady_clock::now();

        // Age cap: a job older than ~1.5 frame intervals at pickup is from a
        // stall; encoding it would read a swapchain slot dangerously close
        // to its next writer (3-deep chain = ~2 intervals of safety from
        // Present). Drop it and take the next fresh frame instead.
        auto const frameIntervalNs
            = std::chrono::nanoseconds((int64_t)(1e9 / Settings_Instance()->m_refreshRate));
        if (workStart - job.enqueueTime > frameIntervalNs + frameIntervalNs / 2) {
            for (int e = 0; e < 2; ++e) {
                if (job.waitFds[e] >= 0) {
                    close(job.waitFds[e]);
                }
            }
            m_ageDropped++;
            continue;
        }

        // Bounded gate on the writer's fences, BEFORE the encoder lock: the
        // render submission waits on these semaphores with an unbounded
        // fence wait while holding m_encMutex, and Present's rebuild paths
        // take the same mutex, so a writer fence that never signals would
        // hang SteamVR's frame loop through the very worker built to keep
        // it free. sync_file fds are pollable: wait here with a cap, and on
        // timeout close the fds and present unguarded (the guard sleep
        // still applies) instead of wedging. After enough consecutive
        // timeouts stop arming the wait entirely, so a permanently wedged
        // writer degrades the stream once instead of taxing every frame.
        if (job.waitFds[0] >= 0 || job.waitFds[1] >= 0) {
            bool dropFds = false;
            if (m_gpuWaitPollDisarmed) {
                dropFds = true;
            } else {
                struct pollfd pfds[2] = {};
                nfds_t count = 0;
                for (int e = 0; e < 2; ++e) {
                    if (job.waitFds[e] >= 0) {
                        pfds[count].fd = job.waitFds[e];
                        pfds[count].events = POLLIN;
                        ++count;
                    }
                }
                int const capMs = (int)(2 * frameIntervalNs.count() / 1'000'000);
                auto const deadline
                    = std::chrono::steady_clock::now() + std::chrono::milliseconds(capMs);
                bool allSignaled = false;
                while (!allSignaled) {
                    int remainingMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                                          deadline - std::chrono::steady_clock::now()
                    )
                                          .count();
                    if (remainingMs < 0) {
                        break;
                    }
                    if (poll(pfds, count, remainingMs) <= 0) {
                        break;
                    }
                    allSignaled = true;
                    for (nfds_t i = 0; i < count; ++i) {
                        if ((pfds[i].revents & POLLIN) == 0) {
                            allSignaled = false;
                        }
                    }
                }
                if (allSignaled) {
                    m_gpuWaitConsecTimeouts = 0;
                } else {
                    dropFds = true;
                    m_gpuWaitPollTimeouts++;
                    if (++m_gpuWaitConsecTimeouts >= kGpuWaitDisarmThreshold) {
                        m_gpuWaitPollDisarmed = true;
                        Info("Encode worker: writer fences timed out %u frames in a row; "
                             "GPU wait disarmed for this session\n",
                             m_gpuWaitConsecTimeouts);
                    }
                }
            }
            if (dropFds) {
                for (int e = 0; e < 2; ++e) {
                    if (job.waitFds[e] >= 0) {
                        close(job.waitFds[e]);
                        job.waitFds[e] = -1;
                    }
                }
            }
        }

        // Guard sleep for fence-INVISIBLE writers (the compositor's own
        // scene writes carry only detached pre-signaled placeholder fences,
        // so the GPU wait cannot see them). Runs on this worker, so it
        // costs pipeline depth, not compositor-thread budget.
        if (m_presentSleepMs > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(m_presentSleepMs));
        }

        try {
            std::lock_guard<std::mutex> encLock(m_encMutex);

            // Generation check: encoder state mutated (set switch, rebuild,
            // shutdown) after this job was built. Its indices and fences
            // describe the OLD state; presenting them against the new one
            // ships another application's pixels under this job's stamp.
            if (job.generation != m_encGeneration) {
                for (int e = 0; e < 2; ++e) {
                    if (job.waitFds[e] >= 0) {
                        close(job.waitFds[e]);
                    }
                }
                m_staleJobsSkipped++;
                continue;
            }

            // ALVR_REWARP: rotationally reproject the frame from the pose it
            // was rendered with to the newest tracking pose, then stamp it
            // with that pose's timestamp: after the warp the newest pose IS
            // the image's pose. The target must be a real PoseHistory entry
            // so the Rust side finds its timestamp in the head pose queue.
            // Sampled here, after the guard sleep, for maximum freshness.
            uint64_t stampTimestampNs = job.matchedTimestampNs;
            alvr::render::WarpParams warpParams {};
            if (m_rewarp && enc.warpCapable()) {
                auto latest = m_poseHistory->GetLatestPose();

                // Dead-band freeze on the warp target: inside the band the
                // target does not move, so stillness imprints zero tracking
                // noise; real motion blows through instantly. Age-capped so
                // the held stamp never grows stale, and the held target is a
                // real PoseHistory entry, so the stamp stays shippable.
                if (latest && m_rewarpDeadbandUrad > 0) {
                    if (m_rewarpHeldValid) {
                        const FfiQuat& a = m_rewarpHeld.motion.pose.orientation;
                        const FfiQuat& b = latest->motion.pose.orientation;
                        double dot = (double)a.w * b.w + (double)a.x * b.x + (double)a.y * b.y
                            + (double)a.z * b.z;
                        double angUrad = 2.0 * acos(fmin(1.0, fabs(dot))) * 1e6;
                        uint64_t ageNs
                            = latest->targetTimestampNs - m_rewarpHeld.targetTimestampNs;
                        if (angUrad < m_rewarpDeadbandUrad && ageNs < 150'000'000ull) {
                            latest = m_rewarpHeld;
                            m_warpStatHolds++;
                        } else {
                            m_rewarpHeld = *latest;
                        }
                    } else {
                        m_rewarpHeld = *latest;
                        m_rewarpHeldValid = true;
                    }
                }

                FfiViewParams viewParams[2];
                bool viewParamsValid;
                {
                    std::unique_lock<std::mutex> vpLock(m_viewParamsMutex);
                    viewParams[0] = m_viewParams[0];
                    viewParams[1] = m_viewParams[1];
                    viewParamsValid = m_viewParamsValid;
                }

                bool warped = false;
                if (latest && viewParamsValid) {
                    warped = BuildWarpParams(
                        job.renderOrientation,
                        latest->motion.pose.orientation,
                        viewParams[0].fov,
                        viewParams[1].fov,
                        warpParams
                    );
                }
                if (warped) {
                    stampTimestampNs = latest->targetTimestampNs;
                    m_warpStatWarped++;
                } else {
                    m_warpStatPassthrough++;
                }
                if (m_warpStatWarped + m_warpStatPassthrough >= 600) {
                    Info(
                        "rewarp: warped=%u passthrough=%u deadbandHolds=%u\n",
                        m_warpStatWarped,
                        m_warpStatPassthrough,
                        m_warpStatHolds
                    );
                    m_warpStatWarped = 0;
                    m_warpStatPassthrough = 0;
                    m_warpStatHolds = 0;
                }
            }

            enc.present(
                job.leftIdx,
                job.rightIdx,
                stampTimestampNs,
                m_rewarp ? &warpParams : nullptr,
                job.waitFds
            );
        } catch (std::exception const& e) {
            // An exception escaping a thread's start function is
            // std::terminate: vrserver dies. Same recovery as the
            // compositor-thread paths: log, tear the encoder down, go idle
            // until the next client connect. The job's fds may be partially
            // consumed by the throw point, so they are deliberately NOT
            // closed here (a bounded leak beats a double-close of a
            // reused descriptor).
            Error("Encode worker: %s. Tearing down encoder and going idle.\n", e.what());
            std::lock_guard<std::mutex> encLock(m_encMutex);
            // Generation deliberately NOT bumped here: it is written only by
            // the compositor thread (Present reads it lock-free). The null
            // encoder makes any straggler job a harmless fd-closing no-op.
            enc.shutdown();
            m_encoderState = EncoderState::Idle;
        }

        uint64_t costUs = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - workStart
        )
                              .count();
        m_workerCostSumUs += costUs;
        if (costUs > m_workerCostMaxUs) {
            m_workerCostMaxUs = costUs;
        }
        m_workerCostFrames++;
        if (m_workerCostFrames >= 600) {
            Info(
                "worker cost 600f: avgUs=%llu maxUs=%llu dropped=%u stale=%u aged=%u "
                "pollTimeouts=%u\n",
                (unsigned long long)(m_workerCostSumUs / m_workerCostFrames),
                (unsigned long long)m_workerCostMaxUs,
                m_jobsDropped.exchange(0),
                m_staleJobsSkipped,
                m_ageDropped,
                m_gpuWaitPollTimeouts
            );
            m_staleJobsSkipped = 0;
            m_ageDropped = 0;
            m_gpuWaitPollTimeouts = 0;
            m_workerCostSumUs = 0;
            m_workerCostMaxUs = 0;
            m_workerCostFrames = 0;
        }
    }
}

void OvrDirectModeComponent::CreateSwapTextureSet(
    uint32_t unPid,
    const SwapTextureSetDesc_t* pSwapTextureSetDesc,
    SwapTextureSet_t* pOutSwapTextureSet
) {
    Info(
        "CreateSwapTextureSet pid=%d Format=%d %dx%d SampleCount=%d\n",
        unPid,
        pSwapTextureSetDesc->nFormat,
        pSwapTextureSetDesc->nWidth,
        pSwapTextureSetDesc->nHeight,
        pSwapTextureSetDesc->nSampleCount
    );

    ProcessResource* processResource = new ProcessResource();
    processResource->textDesc = *pSwapTextureSetDesc;
    processResource->pid = unPid;
    m_swapchainIndices[processResource] = 0;

    {
        auto pid = getpid();
        Info("VrServer PID %d\n", pid);
    }

    uint32_t usageFlags = static_cast<uint32_t>(
        vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eSampled |
        vk::ImageUsageFlagBits::eInputAttachment
    );

    // Allocate the format SteamVR asked for. The read side normalizes to
    // UNORM either way, and substituting UNORM here changes nothing visibly:
    // the chronic darkness in SteamVR-drawn content does not follow the
    // requested format.
    for (int i = 0; i < 3; i++) {
        vr::SharedTextureHandle_t myHandle = 0;
        bool success = vr::VRIPCResourceManager()->NewSharedVulkanImage(
            pSwapTextureSetDesc->nFormat,
            pSwapTextureSetDesc->nWidth,
            pSwapTextureSetDesc->nHeight,
            true,
            false,
            true,
            1,
            1,
            0, // Change creation flags if changed in renderer. Otherwise, the image may not be usable in the renderer.
            usageFlags, // Change usage flags if changed in renderer. Otherwise, the image may not be usable in the renderer.
            &myHandle
        );

        uint64_t ipcHandle = 0;
        vr::VRIPCResourceManager()->RefResource(myHandle, &ipcHandle);

        if (!success) {
            Error("VRCIPCResourceManager: Failed to create shared texture\n");
            CleanupProcessResource(processResource);
            return;
        }

        int fd = 0;
        auto ret = vr::VRIPCResourceManager()->ReceiveSharedFd(ipcHandle, &fd);
        if (ret == false) {
            Error("Failed to get fd for texture\n");
            vr::VRIPCResourceManager()->UnrefResource(myHandle);
            CleanupProcessResource(processResource);
            return;
        }

        processResource->fds[i] = fd;
        processResource->sharedHandles[i] = myHandle;

        m_handleMap.insert(
            std::make_pair(processResource->sharedHandles[i], std::make_pair(processResource, i))
        );
        pOutSwapTextureSet->rSharedTextureHandles[i] = myHandle;
        Info("Created Texture %d %p\n", i, processResource->sharedHandles[i]);
    }

    // Allocate a shared IPC semaphore per texture set. Currently unused:
    // nothing imports, waits on, or signals it.
    vr::SharedTextureHandle_t sharedSemaphoreHandle = 0;
    bool semaphoreCreated
        = vr::VRIPCResourceManager()->NewSharedVulkanSemaphore(false, &sharedSemaphoreHandle);
    if (!semaphoreCreated) {
        Error("VRCIPCResourceManager: Failed to create shared semaphore\n");
        CleanupProcessResource(processResource);
        return;
    }

    uint64_t semaphoreIpcHandle = 0;
    vr::VRIPCResourceManager()->RefResource(sharedSemaphoreHandle, &semaphoreIpcHandle);

    int semaphoreFd = -1;
    auto semaphoreRet = vr::VRIPCResourceManager()->ReceiveSharedFd(semaphoreIpcHandle, &semaphoreFd);
    if (!semaphoreRet) {
        Error("Failed to get fd for shared semaphore\n");
        vr::VRIPCResourceManager()->UnrefResource(sharedSemaphoreHandle);
        CleanupProcessResource(processResource);
        return;
    }

    processResource->sharedSemaphoreHandle = sharedSemaphoreHandle;
    processResource->sharedSemaphoreFd = semaphoreFd;
    Info("Created shared semaphore %p\n", processResource->sharedSemaphoreHandle);
}

/** Used to textures created using CreateSwapTextureSet.  Only one of the set's handles needs to be
 * used to destroy the entire set. */
void OvrDirectModeComponent::DestroySwapTextureSet(vr::SharedTextureHandle_t sharedTextureHandle) {
    Info("DestroySwapTextureSet %p\n", sharedTextureHandle);

    m_presentMutex.lock();
    auto id = m_handleMap.find(sharedTextureHandle);
    if (id != m_handleMap.end()) {
        ProcessResource* p = id->second.first;
        CleanupProcessResource(p);
    } else {
        Debug("Requested to destroy not managing texture. handle:%p\n", sharedTextureHandle);
    }
    m_presentMutex.unlock();
}

/** Used to purge all texture sets for a given process. */
void OvrDirectModeComponent::DestroyAllSwapTextureSets(uint32_t unPid) {
    Info("DestroyAllSwapTextureSets pid=%d\n", unPid);
    
    m_presentMutex.lock();
    std::vector<ProcessResource*> resourcesToDestroy;
    for (auto it = m_handleMap.begin(); it != m_handleMap.end(); ++it) {
        if (it->second.first->pid == unPid && it->second.second == 0) {
            resourcesToDestroy.push_back(it->second.first);
        }
    }
    for (auto* p : resourcesToDestroy) {
        CleanupProcessResource(p);
    }
    m_presentMutex.unlock();
}

void OvrDirectModeComponent::CleanupProcessResource(ProcessResource* processResource) {
    for (int i = 0; i < 3; i++) {
        if (processResource->sharedHandles[i]) {
            vr::VRIPCResourceManager()->UnrefResource(processResource->sharedHandles[i]);
            m_handleMap.erase(processResource->sharedHandles[i]);
            processResource->sharedHandles[i] = 0;
        }
        if (processResource->fds[i] >= 0) {
            close(processResource->fds[i]);
            processResource->fds[i] = -1;
        }
    }

    if (processResource->sharedSemaphoreHandle) {
        vr::VRIPCResourceManager()->UnrefResource(processResource->sharedSemaphoreHandle);
        processResource->sharedSemaphoreHandle = 0;
    }
    if (processResource->sharedSemaphoreFd >= 0) {
        close(processResource->sharedSemaphoreFd);
        processResource->sharedSemaphoreFd = -1;
    }

    m_swapchainIndices.erase(processResource);
    delete processResource;
}

/** After Present returns, calls this to get the next index to use for rendering. */
void OvrDirectModeComponent::GetNextSwapTextureSetIndex(
    vr::SharedTextureHandle_t sharedTextureHandles[2], uint32_t (*pIndices)[2]
) {
    Debug("OvrDirectModeComponent::GetNextSwapTextureSetIndex");

    m_presentMutex.lock();
    for (int eye = 0; eye < 2; eye++) {
        auto it = m_handleMap.find(sharedTextureHandles[eye]);
        if (it == m_handleMap.end()) {
            continue;
        }
        auto& idx = m_swapchainIndices[it->second.first];
        idx = (idx + 1) % 3;
        if (pIndices) {
            (*pIndices)[eye] = idx;
        }
    }
    m_presentMutex.unlock();
}

/** Call once per layer to draw for this frame.  One shared texture handle per eye.  Textures must
 * be created using CreateSwapTextureSet and should be alternated per frame.  Call Present once all
 * layers have been submitted. */
void OvrDirectModeComponent::SubmitLayer(const SubmitLayerPerEye_t (&perEye)[2]) {
    m_presentMutex.lock();

    auto pPose = &perEye[0].mHmdPose; // TODO: are both poses the same? Name HMD suggests yes.

    if (m_submitLayer == 0) {
        // Detect FrameIndex of submitted frame by pPose.
        // This is important part to achieve smooth headtracking.
        // We search for history of TrackingInfo and find the TrackingInfo which have nearest matrix
        // value.

        float matchDiff = 0;
        auto pose = m_poseHistory->GetBestPoseMatch(*pPose, &matchDiff);

        // Temporary 1 Hz diagnostics for the head tracking drag: how exact
        // the rotation match is and how old the chosen entry runs against
        // the newest tracking sample. The client can only look up the last
        // 128 sent poses, so ages beyond that window are guaranteed client
        // misses. One aggregated line per second, nothing per frame.
        constexpr uint64_t kStatWindowNs = 1'000'000'000;
        if (auto latest = m_poseHistory->GetLatestPose()) {
            if (m_matchStatWindowStartNs == 0) {
                m_matchStatWindowStartNs = latest->targetTimestampNs;
            }
            m_matchStatSamples++;
            if (pose) {
                if (matchDiff == 0.f) {
                    m_matchStatExact++;
                }
                uint64_t age = latest->targetTimestampNs - pose->targetTimestampNs;
                if (age > m_matchStatMaxAgeNs) {
                    m_matchStatMaxAgeNs = age;
                }
            } else {
                m_matchStatMisses++;
            }
            // The prediction-time field SteamVR fills and nothing reads: how
            // far ahead of now the submitted pose claims to be predicted.
            // Logged against the match ages to decide whether it can replace
            // matching outright (it names the render pose's time directly).
            float predS = perEye[0].flHmdPosePredictionTimeInSecondsFromNow;
            if (m_predTimeCount == 0) {
                m_predTimeMinS = predS;
                m_predTimeMaxS = predS;
            } else {
                m_predTimeMinS = fminf(m_predTimeMinS, predS);
                m_predTimeMaxS = fmaxf(m_predTimeMaxS, predS);
            }
            m_predTimeSumS += predS;
            m_predTimeCount++;
            if (latest->targetTimestampNs - m_matchStatWindowStartNs >= kStatWindowNs) {
                Info(
                    "PoseMatch 1s: n=%u exact=%u miss=%u maxAgeMs=%.1f predMs mean=%.2f "
                    "min=%.2f max=%.2f\n",
                    m_matchStatSamples,
                    m_matchStatExact,
                    m_matchStatMisses,
                    (double)m_matchStatMaxAgeNs / 1e6,
                    m_predTimeCount ? m_predTimeSumS * 1e3 / m_predTimeCount : 0.0,
                    (double)m_predTimeMinS * 1e3,
                    (double)m_predTimeMaxS * 1e3
                );
                m_matchStatWindowStartNs = latest->targetTimestampNs;
                m_matchStatSamples = 0;
                m_matchStatExact = 0;
                m_matchStatMisses = 0;
                m_matchStatMaxAgeNs = 0;
                m_predTimeSumS = 0.0;
                m_predTimeCount = 0;
            }
        }

        if (pose) {
            // found the frameIndex
            m_prevTargetTimestampNs = m_targetTimestampNs;
            m_targetTimestampNs = pose->targetTimestampNs;

            m_prevFramePoseRotation = m_framePoseRotation;
            m_framePoseRotation.x = pose->motion.pose.orientation.x;
            m_framePoseRotation.y = pose->motion.pose.orientation.y;
            m_framePoseRotation.z = pose->motion.pose.orientation.z;
            m_framePoseRotation.w = pose->motion.pose.orientation.w;
        } else {
            m_targetTimestampNs = 0;
            m_framePoseRotation = HmdQuaternion_Init(0.0, 0.0, 0.0, 0.0);
        }
    }
    if (m_submitLayer < MAX_LAYERS) {
        m_submitLayers[m_submitLayer][0] = perEye[0];
        m_submitLayers[m_submitLayer][1] = perEye[1];
        m_submitLayer++;
    } else {
        Warn("Too many layers submitted!\n");
    }

    m_presentMutex.unlock();
}

/** Submits queued layers for display. */
void OvrDirectModeComponent::Present(vr::SharedTextureHandle_t syncTexture) {
    m_submitLayer = 0;

    switch (m_encoderState.load()) {
    case EncoderState::RebuildRequested:
        Info("Rebuilding encoder from negotiated settings\n");
        DrainPendingJob();
        {
            std::lock_guard<std::mutex> encLock(m_encMutex);
            m_encGeneration++;
            enc.shutdown();
        }
        layer0Texts.fill(0);
        m_encoderState = EncoderState::Streaming;
        break;
    case EncoderState::ShutdownRequested:
        DrainPendingJob();
        {
            std::lock_guard<std::mutex> encLock(m_encMutex);
            m_encGeneration++;
            enc.shutdown();
        }
        layer0Texts.fill(0);
        m_encoderState = EncoderState::Idle;
        return;
    // Before a client connects there are no negotiated settings, so building
    // an encoder here would use wrong defaults and fail noisily. Do nothing
    // until streaming starts.
    case EncoderState::Idle:
        return;
    case EncoderState::Streaming:
        break;
    }

    std::optional<u32> leftIdx;
    std::optional<u32> rightIdx;

    for (u32 i = 0; i < layer0Texts.size(); ++i) {
        if (layer0Texts[i] == m_submitLayers[0][0].hTexture)
            leftIdx = i;

        if (layer0Texts[i] == m_submitLayers[0][1].hTexture)
            rightIdx = i;
    }

    if (!leftIdx.has_value() || !rightIdx.has_value()) {
        auto leftIt = m_handleMap.find(m_submitLayers[0][0].hTexture);
        auto rightIt = m_handleMap.find(m_submitLayers[0][1].hTexture);

        if (leftIt == m_handleMap.end() || rightIt == m_handleMap.end()) {
            Error(
                "Textures not found in handle map %llu, %llu\n",
                m_submitLayers[0][0].hTexture,
                m_submitLayers[0][1].hTexture
            );
            return;
        }

        // Everything below rebuilds encoder state: drop the pending job,
        // wait out the worker's in-flight frame via the mutex, and bump the
        // generation so a job the worker already dequeued (invisible to the
        // drain) gets discarded at its generation check instead of
        // presenting stale indices against the rebuilt state.
        DrainPendingJob();
        std::lock_guard<std::mutex> encLock(m_encMutex);
        m_encGeneration++;

        std::array<int, 6> fds;

        // Vulkan takes ownership of an fd when it imports it, so the copies kept
        // in the handle map are only good for a single import. Hand it a dup
        // every time or the next rebuild fails with InvalidExternalHandle.
        for (u32 i = 0; i < 3; ++i) {
            layer0Texts[i] = leftIt->second.first->sharedHandles[i];
            fds[i] = dup(leftIt->second.first->fds[i]);
        }
        for (u32 i = 0; i < 3; ++i) {
            layer0Texts[i + 3] = rightIt->second.first->sharedHandles[i];
            fds[i + 3] = dup(rightIt->second.first->fds[i]);
        }

        for (u32 i = 0; i < fds.size(); ++i) {
            if (fds[i] == -1) {
                Error("Could not duplicate texture fd %u, skipping encoder setup\n", i);
                for (u32 j = 0; j < fds.size(); ++j) {
                    if (fds[j] != -1)
                        close(fds[j]);
                }
                layer0Texts.fill(0);
                return;
            }
        }

        // Keep a second dup of each texture fd for frame-time fence queries;
        // Vulkan consumes the import dups below. Replaced wholesale on every
        // texture-set switch, so stale sets never accumulate.
        for (u32 i = 0; i < fds.size(); ++i) {
            if (m_syncFds[i] >= 0) {
                close(m_syncFds[i]);
            }
            m_syncFds[i] = dup(fds[i]);
        }

        auto const& settings = Settings_Instance();

        // Hopefully it's the same for both eyes in a layer
        auto& desc = leftIt->second.first->textDesc;

        // SteamVR switches between applications' swap texture sets constantly
        // (dashboard, Home, overlays). When the new set matches the current
        // pipeline's extent and format, re-import just the input images
        // instead of tearing down the renderer and the encoder; the full
        // rebuild below is only for genuine renegotiation.
        vk::Extent2D const newEyeExtent { .width = desc.nWidth, .height = desc.nHeight };
        if (enc.inputsCompatible(newEyeExtent, (vk::Format)(VkFormat)desc.nFormat)) {
            // swapInputs destroys the old imports before creating the new
            // ones, so a throw partway leaves no usable input set. Same
            // recovery as the rebuild below: log, tear down, stay idle.
            try {
                enc.swapInputs(
                    std::array<int, 6> { fds[0], fds[1], fds[2], fds[3], fds[4], fds[5] }
                );
            } catch (std::exception const& e) {
                Error(
                    "Could not swap input textures: %s. Staying idle until the next client "
                    "connect.\n",
                    e.what()
                );
                enc.shutdown();
                layer0Texts.fill(0);
                m_encoderState = EncoderState::Idle;
            }
            // Same contract as the rebuild below: this frame is skipped and
            // the refreshed layer0Texts match from the next one on.
            return;
        }

        alvr::render::RendererCreateInfo rendererCI {
            .format = (vk::Format)(VkFormat)desc.nFormat,
            .inputEyeExtent {
                .width = desc.nWidth,
                .height = desc.nHeight,
            },
            .outputExtent {
                // Stream resolution, not the render-target hint: the encoder
                // and the client's foveation decompression both contract on
                // m_renderWidth/Height, and SteamVR's recommended target can
                // legitimately differ (supersampling). Feeding the hint here
                // skews every foveation ratio by the quotient.
                .width = settings->m_renderWidth,
                .height = settings->m_renderHeight,
            },
            .inputImgFds = fds,
        };

        // Renderer and encoder setup can throw (Vulkan import, VAAPI). An
        // exception escaping Present kills vrserver, so log and stay idle.
        try {
            enc.createImages(rendererCI);
            enc.initEncoding();
        } catch (std::exception const& e) {
            Error(
                "Could not set up the encoder: %s. Staying idle until the next client connect.\n",
                e.what()
            );
            enc.shutdown();
            layer0Texts.fill(0);
            m_encoderState = EncoderState::Idle;
        }

        // We'll get em next time
        return;
    }

    // TODO: Merge layers or something

    // A frame stamped with no matched pose reaches the client as timestamp
    // zero. The client cannot look that up in its view params queue and falls
    // back to the previous frame's pose as the reprojection reference, which
    // shows this frame head locked for a beat. Skipping is strictly better:
    // the client keeps reprojecting the previous frame against a correct
    // reference. This only happens before tracking starts, because the pose
    // buffer never drains once it has been fed.
    if (m_targetTimestampNs == 0) {
        return;
    }

    m_lastLeftSyncIdx = (int)leftIdx.value();
    m_lastRightSyncIdx = (int)rightIdx.value();

    // Sample the fence state at TRUE Present time, before the probe sleep:
    // who does the current write fence belong to, and is it still executing?
    // realActive counts fences pending right now, the number that decides
    // whether a Present-time GPU wait can replace the sleep entirely.
    for (u32 idx : { leftIdx.value(), rightIdx.value() }) {
        if (m_syncFds[idx] < 0) {
            continue;
        }
        auto kind = queryFenceKind(m_syncFds[idx], m_fenceDriverName, sizeof(m_fenceDriverName));
        if (kind == FenceKind::Stub) {
            m_fenceStatStub++;
        } else if (kind == FenceKind::Real) {
            m_fenceStatReal++;
        } else if (kind == FenceKind::RealActive) {
            m_fenceStatRealActive++;
        }
    }

    // GPU-side write wait: export each eye's read-wait sync file and hand it
    // to the renderer, whose submission waits on it at the transfer stage.
    // Exact (the copy starts the moment the writer's fence signals, however
    // long that takes) where the sleep below is a guess. When both exports
    // arm, the CPU sleep is skipped entirely.
    int waitFds[2] = { -1, -1 };
    if (m_gpuWait) {
        u32 const eyeSlot[2] = { leftIdx.value(), rightIdx.value() };
        for (int e = 0; e < 2; ++e) {
            if (m_syncFds[eyeSlot[e]] >= 0) {
                waitFds[e] = exportReadSyncFile(m_syncFds[eyeSlot[e]]);
            }
        }
    }
    bool const gpuWaitArmed = waitFds[0] >= 0 && waitFds[1] >= 0;
    if (!gpuWaitArmed) {
        for (int e = 0; e < 2; ++e) {
            if (waitFds[e] >= 0) {
                close(waitFds[e]);
                waitFds[e] = -1;
            }
        }
        m_gpuWaitFallback++;
    } else {
        m_gpuWaitArmed++;
    }

    // No CPU wait here anymore: the guard sleep (for fence-invisible
    // writers) and the render/encode chain both live on the encode worker.
    // The dormant adaptive-wait experiment was removed with them; it is
    // superseded by the renderer's GPU-side fence wait (git has it).

    m_fenceStatFrames++;
    if (m_fenceStatWindowStartNs == 0) {
        m_fenceStatWindowStartNs = m_targetTimestampNs;
    }
    if (m_targetTimestampNs - m_fenceStatWindowStartNs >= kFenceStatWindowNs) {
        Info(
            "FenceWait 1s (guard %dms): n=%u stub=%u real=%u realActive=%u postReal=%u drv=%s\n",
            m_presentSleepMs,
            m_fenceStatFrames,
            m_fenceStatStub,
            m_fenceStatReal,
            m_fenceStatRealActive,
            m_fenceStatPostReal,
            m_fenceDriverName
        );
        m_fenceStatWindowStartNs = m_targetTimestampNs;
        m_fenceStatFrames = 0;
        m_fenceStatStub = 0;
        m_fenceStatReal = 0;
        m_fenceStatRealActive = 0;
        m_fenceStatPostReal = 0;
    }

    // Compositor-thread cost probe: with the encode worker in place this
    // should read well under a millisecond. This is the cost SteamVR's frame
    // loop actually pays for us.
    auto const presentCostStart = std::chrono::steady_clock::now();

    // Hand the frame to the encode worker: warp target selection, the guard
    // sleep, render, encode and send all happen there. The mailbox is a
    // single latest-wins slot; a stale pending job means the worker fell
    // behind a full frame, and the newest frame replaces it (drop counted,
    // fds closed).
    FrameJob job {};
    job.leftIdx = leftIdx.value();
    job.rightIdx = rightIdx.value();
    job.matchedTimestampNs = m_targetTimestampNs;
    job.renderOrientation = m_framePoseRotation;
    job.waitFds[0] = waitFds[0];
    job.waitFds[1] = waitFds[1];
    // Written only on this compositor thread, so the unlocked read is its
    // own most recent write.
    job.generation = m_encGeneration;
    job.enqueueTime = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
        if (m_workerExit) {
            // Teardown already drained; do not repopulate a dying mailbox.
            for (int e = 0; e < 2; ++e) {
                if (job.waitFds[e] >= 0) {
                    close(job.waitFds[e]);
                }
            }
            return;
        }
        if (m_jobPending) {
            for (int e = 0; e < 2; ++e) {
                if (m_job.waitFds[e] >= 0) {
                    close(m_job.waitFds[e]);
                }
            }
            m_jobsDropped++;
        }
        m_job = job;
        m_jobPending = true;
    }
    m_jobCv.notify_one();

    uint64_t costUs = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - presentCostStart
    )
                          .count();
    m_presentCostSumUs += costUs;
    if (costUs > m_presentCostMaxUs) {
        m_presentCostMaxUs = costUs;
    }
    m_presentCostFrames++;
    if (m_presentCostFrames >= 600) {
        Info(
            "present cost 600f: avgUs=%llu maxUs=%llu gpuWaitArmed=%u fallback=%u\n",
            (unsigned long long)(m_presentCostSumUs / m_presentCostFrames),
            (unsigned long long)m_presentCostMaxUs,
            m_gpuWaitArmed,
            m_gpuWaitFallback
        );
        m_presentCostSumUs = 0;
        m_presentCostMaxUs = 0;
        m_presentCostFrames = 0;
        m_gpuWaitArmed = 0;
        m_gpuWaitFallback = 0;
    }
}

void OvrDirectModeComponent::PostPresent(const Throttling_t* pThrottling) {
    // Prop_Hmd_SupportsAppThrottling_Bool is set (HMD.cpp), so SteamVR sends
    // throttle hints here and adjusts its pose prediction assuming we honor
    // them. Ignoring them while it throttles an app means poses predicted
    // for a cadence we are not keeping, which reprojects the world against
    // head motion. Log changes and stretch the vsync grid accordingly.
    uint32_t throttleFrames = 0;
    if (pThrottling != nullptr) {
        throttleFrames = pThrottling->nFramesToThrottle;
        if (pThrottling->nFramesToThrottle != m_lastThrottleFrames
            || pThrottling->nAdditionalFramesToPredict != m_lastPredictFrames) {
            m_lastThrottleFrames = pThrottling->nFramesToThrottle;
            m_lastPredictFrames = pThrottling->nAdditionalFramesToPredict;
            Info(
                "Throttling: framesToThrottle=%u additionalFramesToPredict=%u\n",
                pThrottling->nFramesToThrottle,
                pThrottling->nAdditionalFramesToPredict
            );
        }
    }
    // A direct mode driver paces the compositor: vrserver starts the next
    // frame when we send VsyncEvent. Firing it immediately lets the
    // compositor free-run as fast as encoding completes, which floods the
    // client's decoder (frame queue overflow, decoder saturation drops) and
    // starves the rest of the desktop. PostPresent exists exactly so the
    // driver can hold this thread until vsync, so pace it to the negotiated
    // refresh rate.
    using std::chrono::nanoseconds;
    using std::chrono::steady_clock;

    auto const frameInterval
        = nanoseconds((int64_t)(1e9 / Settings_Instance()->m_refreshRate));
    auto const now = steady_clock::now();

    if (m_nextVsync == steady_clock::time_point {}) {
        // First frame: the default-constructed epoch would otherwise fall
        // into the catch-up loop below and iterate once per frame interval
        // of machine uptime, inflating m_skippedVsyncs with nonsense.
        m_nextVsync = now + frameInterval;
    } else if (m_nextVsync > now + frameInterval) {
        // Clock went backwards: resynchronize.
        m_nextVsync = now + frameInterval;
    } else {
        // Advance along the fixed grid, skipping any ticks we missed. A loop
        // that runs slightly over budget every frame (heavy app scenes) must
        // lose whole ticks cleanly instead of accumulating debt and then
        // snapping, which showed up as a violent burst-pause-burst cadence
        // and ~50ms of queueing latency in VRChat. Every skipped tick is
        // reported to the compositor via GetFrameTiming as a dropped frame,
        // so its pose prediction tracks the REAL display cadence; without
        // that report poses are predicted one vsync out while frames display
        // several later, which reprojects the world opposite to head motion.
        m_nextVsync += frameInterval;
        while (m_nextVsync <= now) {
            m_nextVsync += frameInterval;
            m_skippedVsyncs++;
        }
    }

    // Honor SteamVR's throttle: hold the next vsync for the requested number
    // of additional frames, so the cadence we signal matches the cadence its
    // pose prediction assumes.
    m_nextVsync += frameInterval * throttleFrames;

    // Running start: wake a phase lead BEFORE the frame target and declare the
    // vsync at that announce point rather than at the tick. VsyncEvent takes a
    // time offset in seconds (openvr_driver.h names the parameter
    // vsyncTimeOffsetSeconds), so declaring early gives the grid that
    // WaitGetPoses release and pose prediction are computed against a head
    // start on the real deadline. Sleeping to the tick and declaring
    // VsyncEvent(0.0) advertises zero headroom, which measured as heavy apps
    // locked to half rate on this rig; a 2 ms lead measured as a full 72.
    // ALVR_RUNNING_START_US tunes the phase; 0 restores the exact-tick
    // behavior.
    auto const runningStart = std::chrono::microseconds(m_runningStartUs);
    auto const announce = m_nextVsync - runningStart;
    std::this_thread::sleep_until(announce);

    // Late fence sample: a real (non-stub) write fence appearing HERE but not
    // at the top of Present means the writer submits its GPU work after
    // Present, and any Present-time wait is sampling too early by design.
    for (int idx : { m_lastLeftSyncIdx, m_lastRightSyncIdx }) {
        if (idx >= 0 && m_syncFds[idx] >= 0) {
            if (queryFenceKind(m_syncFds[idx], nullptr, 0) == FenceKind::Real) {
                m_fenceStatPostReal++;
            }
        }
    }

    if (m_sendVsyncEvents) {
        // The offset self-corrects wake jitter: it goes slightly negative when
        // we oversleep, which reads as "the vsync just happened".
        double offset = std::chrono::duration_cast<std::chrono::duration<double>>(
                            announce - steady_clock::now()
        )
                            .count();
        vr::VRServerDriverHost()->VsyncEvent(offset);
    }
}

void OvrDirectModeComponent::GetFrameTiming(vr::DriverDirectMode_FrameTiming* pFrameTiming) {
    if (pFrameTiming == nullptr
        || pFrameTiming->m_nSize < sizeof(vr::DriverDirectMode_FrameTiming)) {
        return;
    }

    // Reporting skipped vsyncs as m_nNumDroppedFrames LOOKED like the honest
    // cadence feedback, but the compositor treats it as a GPU-overload signal
    // and enables interleaved reprojection: half rate -> our grid then skips
    // a tick every frame -> we report another drop -> throttled forever.
    // Measured: Home locked to 35fps with the report in, 72 without. So:
    // present counts stay nominal, and the skip counter is consumed here
    // without being reported until a non-throttling channel for it exists.
    (void)m_skippedVsyncs.exchange(0);
    pFrameTiming->m_nNumFramePresents = 1;
    pFrameTiming->m_nNumMisPresented = 0;
    pFrameTiming->m_nNumDroppedFrames = 0;
    // Cleared per the SDK note: these bits overlap the throttle mask and
    // must not be echoed back unhandled.
    pFrameTiming->m_nReprojectionFlags = 0;
}
