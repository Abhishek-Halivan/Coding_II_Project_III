#include "EdgeDetector.h"
#include <opencv2/imgproc.hpp>

EdgeDetector::EdgeDetector()
    : clahe(cv::createCLAHE(2.5, cv::Size(8, 8))) {
}

cv::Mat EdgeDetector::enhance(const cv::Mat& grayImage) const {
    cv::Mat enhancedGray;
    clahe->apply(grayImage, enhancedGray);
    return enhancedGray;
}

cv::Mat EdgeDetector::detectEdges(const cv::Mat& enhancedGray) const {
    cv::Mat blurred;
    cv::GaussianBlur(enhancedGray, blurred, cv::Size(3, 3), 0.0);
    
    cv::Mat edges;
    cv::Canny(blurred, edges, canny_low, canny_high);
    return edges;
}

double EdgeDetector::getEdgeDensityRatio(const cv::Mat& edges, const cv::Rect& region) const {
    const cv::Mat regionEdges = edges(region);
    const double edgeRatio = static_cast<double>(cv::countNonZero(regionEdges)) /
                             static_cast<double>(region.area());
    return edgeRatio;
}
