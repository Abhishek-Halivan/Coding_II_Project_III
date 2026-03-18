#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <utility>

// Utilities for combining detections from multiple detectors and computing detection metrics.
// Fuses Haar cascade and DNN detections to achieve higher accuracy and lower false positive rate.
// Implements temporal smoothing and IoU-based detection matching.
class DetectionFusion {
public:
    // Compute Intersection over Union (IoU) between two rectangles.
    // Formula: IoU = Intersection Area / Union Area
    // Returns 0.0 for non-overlapping rectangles, 1.0 for identical rectangles.
    // Used to determine if two detectors found the same face (typically threshold ~0.3).
    static double computeIoU(const cv::Rect& a, const cv::Rect& b);

    // Temporally smooth a rectangle position using weighted averaging.
    // alpha: blending factor (0.0 = full current, 1.0 = full previous).
    // Reduces sudden jumps in bounding box position between frames.
    // Returns smoothed rectangle at interpolated position.
    static cv::Rect smoothRect(const cv::Rect& prev, const cv::Rect& curr, double alpha);

    // Merge detections from Haar cascade and DNN to produce confidence-scored candidates.
    // Strategy:
    //   1. If both detectors agree (high IoU), boost confidence to fusion_base_confidence
    //   2. If only Haar detects, assign haar_only_confidence
    //   3. If only DNN detects, assign dnn_only_confidence
    // Returns vector of (rectangle, confidence_score) pairs for downstream filtering.
    static std::vector<std::pair<cv::Rect, double>> fuseDetections(
        const std::vector<cv::Rect>& haarFaces,
        const std::vector<cv::Rect>& dnnFaces
    );

private:
    static constexpr double iou_threshold = 0.3;
    static constexpr double haar_only_confidence = 0.7;
    static constexpr double dnn_only_confidence = 0.75;
    static constexpr double fusion_base_confidence = 0.85;
};
