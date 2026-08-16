#include "PoseHistory.h"
#include "Logger.h"
#include "Utils.h"
#include "include/openvr_math.h"
#include <cstdlib>
#include <mutex>
#include <optional>

PoseHistory::PoseHistory() {
    const char* windowMs = getenv("ALVR_POSE_MATCH_WINDOW_MS");
    if (windowMs != nullptr) {
        m_matchWindowNs = (uint64_t)atoi(windowMs) * 1000000ull;
    }
}

void PoseHistory::OnPoseUpdated(uint64_t targetTimestampNs, FfiDeviceMotion motion) {
    // Put pose history buffer
    TrackingHistoryFrame history;
    history.targetTimestampNs = targetTimestampNs;
    history.motion = motion;

    HmdMatrix_QuatToMat(
        motion.pose.orientation.w,
        motion.pose.orientation.x,
        motion.pose.orientation.y,
        motion.pose.orientation.z,
        &history.rotationMatrix
    );

    std::unique_lock<std::mutex> lock(m_mutex);
    if (!m_transformIdentity) {
        vr::HmdMatrix34_t rotation = vrmath::matMul33(m_transform, history.rotationMatrix);
        history.rotationMatrix = rotation;
    }

    if (m_poseBuffer.size() == 0) {
        m_poseBuffer.push_back(history);
    } else {
        if (m_poseBuffer.back().targetTimestampNs != targetTimestampNs) {
            // New track info
            m_poseBuffer.push_back(history);
        }
    }
    // The value should match with the client's MAXIMUM_TRACKING_FRAMES in ovr_context.cpp
    if (m_poseBuffer.size() > 120 * 3) {
        m_poseBuffer.pop_front();
    }
}

std::optional<PoseHistory::TrackingHistoryFrame>
PoseHistory::GetBestPoseMatch(const vr::HmdMatrix34_t& pose, float* outMinDiff) const {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_poseBuffer.empty()) {
        Debug("PoseHistory::GetBestPoseMatch: No pose matched.");
        return {};
    }

    // Search newest first. The buffer holds about 1.7 seconds of tracking, so
    // when the head sweeps back over an orientation it recently held, several
    // entries match the submitted pose equally well. Scanning oldest first
    // returned the oldest of those, stamping the frame with a pose up to a
    // second stale, which reads as a snap while sweeping. Ties now resolve to
    // the most recent entry, and entries older than the search window are not
    // considered at all (a rendered frame's pose is always recent).
    const uint64_t newestTimestampNs = m_poseBuffer.back().targetTimestampNs;
    float minDiff = 100000;
    auto minIt = m_poseBuffer.rbegin();
    for (auto it = m_poseBuffer.rbegin(); it != m_poseBuffer.rend(); ++it) {
        if (m_matchWindowNs != 0 && newestTimestampNs - it->targetTimestampNs > m_matchWindowNs) {
            break;
        }
        float distance = 0;
        // Rotation matrix composes a part of ViewMatrix of TrackingInfo.
        // Be carefull of transpose.
        // And bottom side and right side of matrix should not be compared, because pPose does not
        // contain that part of matrix.
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                distance += pow(it->rotationMatrix.m[j][i] - pose.m[j][i], 2);
            }
        }
        if (minDiff > distance) {
            minIt = it;
            minDiff = distance;
        }
    }
    if (outMinDiff) {
        *outMinDiff = minDiff;
    }
    return *minIt;
}

std::optional<PoseHistory::TrackingHistoryFrame> PoseHistory::GetLatestPose() const {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_poseBuffer.empty()) {
        return {};
    }
    return m_poseBuffer.back();
}

std::optional<PoseHistory::TrackingHistoryFrame> PoseHistory::GetPoseAt(uint64_t timestampNs
) const {
    std::unique_lock<std::mutex> lock(m_mutex);
    for (auto it = m_poseBuffer.rbegin(), end = m_poseBuffer.rend(); it != end; ++it) {
        if (it->targetTimestampNs == timestampNs)
            return *it;
    }

    Debug("PoseHistory::GetPoseAt: No pose matched.");
    return {};
}

void PoseHistory::SetTransform(const vr::HmdMatrix34_t& transform) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_transform = transform;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (transform.m[i][j] != (i == j ? 1 : 0)) {
                m_transformIdentity = false;
                return;
            }
        }
    }
    m_transformIdentity = true;
}
