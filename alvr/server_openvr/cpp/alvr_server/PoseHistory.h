#pragma once

#include "ALVR-common/packet_types.h"
#include "openvr_driver_wrap.h"

#include <list>
#include <mutex>
#include <optional>

class PoseHistory {
public:
    struct TrackingHistoryFrame {
        uint64_t targetTimestampNs;
        FfiDeviceMotion motion;
        vr::HmdMatrix34_t rotationMatrix;
    };

    // Reads ALVR_POSE_MATCH_WINDOW_MS (default 250) for the match search bound.
    PoseHistory();

    void OnPoseUpdated(uint64_t targetTimestampNs, FfiDeviceMotion motion);

    // outMinDiff, when given, receives the squared distance of the winning
    // entry, or stays untouched when the buffer is empty.
    std::optional<TrackingHistoryFrame>
    GetBestPoseMatch(const vr::HmdMatrix34_t& pose, float* outMinDiff = nullptr) const;
    // Return the most recent pose known at the given timestamp
    std::optional<TrackingHistoryFrame> GetPoseAt(uint64_t timestampNs) const;
    // Newest tracking entry, if any.
    std::optional<TrackingHistoryFrame> GetLatestPose() const;

    void SetTransform(const vr::HmdMatrix34_t& transform);

private:
    mutable std::mutex m_mutex;
    std::list<TrackingHistoryFrame> m_poseBuffer;
    // How far back GetBestPoseMatch searches, from the newest entry. 0 = no
    // bound (the full buffer).
    uint64_t m_matchWindowNs = 250000000ull;
    vr::HmdMatrix34_t m_transform
        = { { { 1.0, 0.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0, 0.0 } } };
    bool m_transformIdentity = true;
};
