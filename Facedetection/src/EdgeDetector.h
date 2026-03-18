#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Handles image preprocessing and edge analysis to improve detection quality.
// Enhances low-contrast or backlit images using CLAHE (Contrast-Limited Adaptive Histogram Equalization).
// Detects facial edges (Canny algorithm) to analyze and filter candidate face regions.
class EdgeDetector {
public:
    EdgeDetector();

    // Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization) to grayscale image.
    // Improves local contrast, making faces more visible in challenging lighting conditions.
    // Crucial for detecting faces in backlit or low-light scenarios.
    cv::Mat enhance(const cv::Mat& grayImage) const;

    // Detect edges using Canny edge detector on enhanced grayscale input.
    // Returns binary edge map (white edges on black background).
    // Used for face validation: presence of eye and facial feature edges indicates a real face.
    cv::Mat detectEdges(const cv::Mat& enhancedGray) const;

    // Calculate proportion of edge pixels within a region of interest.
    // Returns ratio of edge pixels to total region area (0.0 to 1.0).
    // Helps filter false positives: faces have moderate edge density; pure edges or blank areas don't.
    double getEdgeDensityRatio(const cv::Mat& edges, const cv::Rect& region) const;

private:
    cv::Ptr<cv::CLAHE> clahe;
    static constexpr int canny_low = 55;
    static constexpr int canny_high = 150;
};
