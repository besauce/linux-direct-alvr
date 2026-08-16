#pragma once
#include "alvr_server/PoseHistory.h"
#include "alvr_server/Utils.h"
#include "openvr_driver.h"

#include "alvr_server/Settings.h"

#include <map>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <vulkan/vulkan.h>

#include "Encoder.hpp"

class OvrDirectModeComponent : public vr::IVRDriverDirectModeComponent {
public:
    OvrDirectModeComponent(
        /* std::shared_ptr<Renderer> pVKRender,  */ std::shared_ptr<PoseHistory> poseHistory
    );
    ~OvrDirectModeComponent();

    void RequestIdr() { enc.requestIdr(); }

    // Called from the event loop thread when negotiated settings arrive.
    // The state change itself is applied on the compositor thread inside
    // Present, which then builds the encoder from the new settings.
    void RequestEncoderReset() { m_encoderState = EncoderState::RebuildRequested; }

    // Called from the event loop thread when the client disconnects. Present
    // tears the encoder down and goes idle until the next connect.
    void RequestEncoderShutdown() { m_encoderState = EncoderState::ShutdownRequested; }

    /** Specific to Oculus compositor support, textures supplied must be created using this method.
     */
    virtual void CreateSwapTextureSet(
        uint32_t unPid,
        const SwapTextureSetDesc_t* pSwapTextureSetDesc,
        SwapTextureSet_t* pOutSwapTextureSet
    );

    /** Used to textures created using CreateSwapTextureSet.  Only one of the set's handles needs to
     * be used to destroy the entire set. */
    virtual void DestroySwapTextureSet(vr::SharedTextureHandle_t sharedTextureHandle);

    /** Used to purge all texture sets for a given process. */
    virtual void DestroyAllSwapTextureSets(uint32_t unPid);

    /** After Present returns, calls this to get the next index to use for rendering. */
    virtual void GetNextSwapTextureSetIndex(
        vr::SharedTextureHandle_t sharedTextureHandles[2], uint32_t (*pIndices)[2]
    );

    /** Call once per layer to draw for this frame.  One shared texture handle per eye.  Textures
     * must be created using CreateSwapTextureSet and should be alternated per frame.  Call Present
     * once all layers have been submitted. */
    virtual void SubmitLayer(const SubmitLayerPerEye_t (&perEye)[2]);

    /** Submits queued layers for display. */
    virtual void Present(vr::SharedTextureHandle_t syncTexture);

    /** Called after Present to allow driver to take more time until vsync after they've
     * successfully acquired the sync texture in Present.*/
    virtual void PostPresent(const Throttling_t* pThrottling);

    /** Reports real presentation cadence (skipped vsyncs) so the compositor's
     * pose prediction matches actual display time at below-full frame rates. */
    virtual void GetFrameTiming(vr::DriverDirectMode_FrameTiming* pFrameTiming);

    // View params from the client (per-eye FOV), needed for the warp's
    // tangent bounds. Called from the event loop thread; read in Present.
    void SetViewParams(const FfiViewParams params[2]);

private:
    std::shared_ptr<PoseHistory> m_poseHistory;

    // Pre-encode rotational reprojection from the frame's render pose to the
    // newest tracking pose. On by default; ALVR_REWARP=0 disables it for A/B
    // runs. Flags read once at construction.
    bool m_rewarp = true;
    // Dead-band hold on the warp target, microradians. 0 = off, the shipped
    // default: held stamps inflate measured latency and chase the prediction
    // horizon up (measured), so the hold stays disabled pending a
    // stamp-age-aware redesign. ALVR_REWARP_DEADBAND_URAD overrides for
    // experiments.
    double m_rewarpDeadbandUrad = 0.0;
    // Write-race guard sleep in Present, ms. ALVR_PRESENT_SLEEP_MS overrides.
    int m_presentSleepMs = 4;
    // GPU-side wait on the writer's dma-buf fences, replacing the sleep when
    // both eyes' exports arm. ALVR_GPU_WAIT=0 disables. Present thread only.
    bool m_gpuWait = true;
    // Running-start phase: the declared vsync grid leads the real frame
    // deadline by this many microseconds. 2000 is the measured default.
    // ALVR_RUNNING_START_US overrides; 0 = old exact-tick behavior.
    int m_runningStartUs = 2000;
    // ALVR_NO_VSYNC_EVENTS=1 stops calling VsyncEvent (with the property
    // cleared in HMD.cpp this reproduces the win32 hardcoded-offset model,
    // the known-working fallback configuration).
    bool m_sendVsyncEvents = true;
    uint32_t m_gpuWaitArmed = 0;
    uint32_t m_gpuWaitFallback = 0;
    // Worker-side bounded gate on the writer fences (see EncodeWorkerLoop):
    // consecutive poll timeouts disarm the GPU wait for the session so a
    // wedged writer degrades the stream instead of taxing every frame.
    static constexpr uint32_t kGpuWaitDisarmThreshold = 30;
    uint32_t m_gpuWaitPollTimeouts = 0;
    uint32_t m_gpuWaitConsecTimeouts = 0;
    bool m_gpuWaitPollDisarmed = false;
    uint64_t m_presentCostSumUs = 0;
    uint64_t m_presentCostMaxUs = 0;
    uint32_t m_presentCostFrames = 0;
    std::mutex m_viewParamsMutex;
    FfiViewParams m_viewParams[2] = {};
    bool m_viewParamsValid = false;
    // Dead-band hold state and 1 Hz warp stats, present thread only.
    PoseHistory::TrackingHistoryFrame m_rewarpHeld {};
    bool m_rewarpHeldValid = false;
    uint32_t m_warpStatWarped = 0;
    uint32_t m_warpStatPassthrough = 0;
    uint32_t m_warpStatHolds = 0;

    // 1 Hz stats over the prediction-time field SteamVR hands us in
    // SubmitLayer and nothing reads today. Instrumentation for deciding
    // whether the field can replace pose matching; it must not drive
    // anything until its clock relationship to the tracking timestamps is
    // measured.
    float m_predTimeMinS = 0.f;
    float m_predTimeMaxS = 0.f;
    double m_predTimeSumS = 0.0;
    uint32_t m_predTimeCount = 0;

    // Resource for each process
    struct ProcessResource {
        vr::SharedTextureHandle_t sharedHandles[3];
        // Cleanup closes any entry >= 0, so unset slots must not read as 0.
        int fds[3] = { -1, -1, -1 };
        vr::SharedTextureHandle_t sharedSemaphoreHandle = 0;
        int sharedSemaphoreFd = -1;
        SwapTextureSetDesc_t textDesc;
        uint32_t pid;
    };
    std::map<vr::SharedTextureHandle_t, std::pair<ProcessResource*, int>> m_handleMap;
    void CleanupProcessResource(ProcessResource* processResource);

    static const int MAX_LAYERS = 10;
    int m_submitLayer;
    SubmitLayerPerEye_t m_submitLayers[MAX_LAYERS][2];
    vr::HmdQuaternion_t m_prevFramePoseRotation;

    // 1 Hz pose match diagnostics, temporary instrumentation. Timestamps and
    // ages are in the tracking clock domain, so no wall clock reads on the
    // present thread.
    uint64_t m_matchStatWindowStartNs = 0;
    uint32_t m_matchStatSamples = 0;
    uint32_t m_matchStatExact = 0;
    uint32_t m_matchStatMisses = 0;
    uint64_t m_matchStatMaxAgeNs = 0;

    // Retained dups of the current texture set's dma-buf fds, indexed like
    // layer0Texts. Vulkan consumes the import dups, so these are the only
    // handles left to query fences on. Replaced wholesale on texture-set
    // switch; -1 means dup failed and the frame-time wait skips that slot.
    std::array<int, 6> m_syncFds { -1, -1, -1, -1, -1, -1 };

    // 1 Hz writer-fence wait diagnostics, temporary instrumentation for the
    // band flicker / head drag investigation.
    uint64_t m_fenceStatWindowStartNs = 0;
    uint32_t m_fenceStatFrames = 0;

    // Fence-kind discrimination: stub vs real write fences on the dma-bufs,
    // sampled at Present (early) and again in PostPresent (late, catches
    // writer work submitted after Present).
    uint32_t m_fenceStatStub = 0;
    uint32_t m_fenceStatReal = 0;
    uint32_t m_fenceStatRealActive = 0;
    uint32_t m_fenceStatPostReal = 0;

    // Vsync ticks skipped by the pacing grid since the last GetFrameTiming,
    // reported to the compositor as dropped frames. Written on the present
    // thread, consumed on whichever thread the compositor queries from.
    std::atomic<uint32_t> m_skippedVsyncs { 0 };

    // Last throttle hints received in PostPresent, for change logging.
    uint32_t m_lastThrottleFrames = 0;
    uint32_t m_lastPredictFrames = 0;
    char m_fenceDriverName[80] = "none";
    int m_lastLeftSyncIdx = -1;
    int m_lastRightSyncIdx = -1;
    vr::HmdQuaternion_t m_framePoseRotation;
    uint64_t m_targetTimestampNs;
    uint64_t m_prevTargetTimestampNs;
    // Track current texture index for each eye to avoid UB with uninitialized data
    std::map<ProcessResource*, uint32_t> m_swapchainIndices;

    std::array<vr::SharedTextureHandle_t, 6> layer0Texts {};

    alvr::Encoder enc;

    // Written by the event loop thread, applied and advanced by the
    // compositor thread at the top of Present.
    enum class EncoderState { Idle, RebuildRequested, Streaming, ShutdownRequested };
    std::atomic<EncoderState> m_encoderState { EncoderState::Idle };

    // Next frame slot for PostPresent's vsync pacing. Only touched from the
    // compositor thread that runs Present/PostPresent.
    std::chrono::steady_clock::time_point m_nextVsync {};

    std::mutex m_presentMutex;

    // ---- Encode worker pipeline ----
    // Present runs on the compositor's frame loop, so it hands each frame to a
    // dedicated worker instead of encoding inline and stalling that loop; the
    // worker owns the guard sleep, the warp target selection, and the
    // render/encode/send chain.
    // Single-slot latest-wins mailbox: at steady state the worker drains
    // faster than frames arrive, and if it ever falls behind, the newest
    // frame replaces the stale pending one (drop counted).
    struct FrameJob {
        uint32_t leftIdx;
        uint32_t rightIdx;
        uint64_t matchedTimestampNs;
        vr::HmdQuaternion_t renderOrientation;
        int waitFds[2];
        // Encoder-state generation this job was built against; the worker
        // discards the job if the generation moved (set switch, rebuild,
        // shutdown) between enqueue and processing.
        uint64_t generation;
        std::chrono::steady_clock::time_point enqueueTime;
    };
    void EncodeWorkerLoop();
    // Drop a pending job (closing its fds). Call before any path that
    // touches the encoder from the compositor thread.
    void DrainPendingJob();

    std::thread m_encodeWorker;
    std::mutex m_jobMutex;
    std::condition_variable m_jobCv;
    FrameJob m_job {};
    bool m_jobPending = false;
    bool m_workerExit = false;
    std::atomic<uint32_t> m_jobsDropped { 0 };
    // Serializes encoder access: the worker's per-frame enc.present against
    // the compositor thread's rebuild/swap/shutdown paths. Never held
    // together with m_jobMutex.
    std::mutex m_encMutex;
    // Incremented (under m_encMutex) by every compositor-thread path that
    // mutates encoder state. Written only by the compositor thread, so it
    // reads its own value lock-free; the worker reads under m_encMutex. A
    // job whose generation is stale gets dropped instead of presenting old
    // indices against rebuilt state.
    uint64_t m_encGeneration = 0;
    uint32_t m_staleJobsSkipped = 0;
    uint32_t m_ageDropped = 0;
    // Worker-side 600-frame cost probe (sleep + render + encode + send).
    uint64_t m_workerCostSumUs = 0;
    uint64_t m_workerCostMaxUs = 0;
    uint32_t m_workerCostFrames = 0;
};
