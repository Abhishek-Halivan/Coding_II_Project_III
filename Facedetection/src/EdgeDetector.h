#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/**
 * Handles edge detection and CLAHE-based contrast enhancement
 */
class EdgeDetector {
public:
    EdgeDetector();

    /**
     * Enhance grayscale image using CLAHE
     */
    cv::Mat enhance(const cv::Mat& grayImage) const;

    /**
     * Detect edges using Canny edge detector
     */
    cv::Mat detectEdges(const cv::Mat& enhancedGray) const;

    /**
     * Calculate edge density ratio in a region
     */
    double getEdgeDensityRatio(const cv::Mat& edges, const cv::Rect& region) const;

private:
    cv::Ptr<cv::CLAHE> clahe;
    static constexpr int canny_low = 55;
    static constexpr int canny_high = 150;
};
