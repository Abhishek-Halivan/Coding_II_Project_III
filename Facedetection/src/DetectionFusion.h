#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <utility>

/**
 * Utilities for computing detection metrics and merging detections from multiple detectors
 */
class DetectionFusion {
public:
    /**
     * Compute Intersection over Union (IoU) between two rectangles
     */
    static double computeIoU(const cv::Rect& a, const cv::Rect& b);

    /**
     * Temporally smooth a rectangle based on previous detection
     * @param prev Previous rectangle position
     * @param curr Current rectangle position
     * @param alpha Smoothing factor (higher = more weight to previous)
     * @return Smoothed rectangle
     */
    static cv::Rect smoothRect(const cv::Rect& prev, const cv::Rect& curr, double alpha);

    /**
     * Merge detections from Haar cascade and DNN detector
     * @param haarFaces Faces detected by Haar cascade
     * @param dnnFaces Faces detected by DNN
     * @return Vector of (rectangle, confidence) pairs
     */
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
