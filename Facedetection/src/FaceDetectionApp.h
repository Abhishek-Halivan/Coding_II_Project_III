#pragma once

#include "FaceDetector.h"
#include "DnnFaceDetector.h"
#include "EdgeDetector.h"
#include <opencv2/core.hpp>
#include <vector>
#include <string>

/**
 * Main face detection application orchestrating all detection and visualization components
 */
class FaceDetectionApp {
public:
    FaceDetectionApp();

    /**
     * Initialize the application
     * @param cascadePath Path to Haar cascade file (optional, checks env var if empty)
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const std::string& cascadePath = "");

    /**
     * Run the main detection loop
     */
    void run();

private:
    FaceDetector haarDetector;
    DnnFaceDetector dnnDetector;
    EdgeDetector edgeDetector;

    // Detection thresholds
    static constexpr double detection_scale = 0.5;
    static constexpr double temporal_alpha = 0.7;
    static constexpr double dnn_confidence_threshold = 0.6;
    static constexpr double min_edge_ratio = 0.025;
    static constexpr double max_edge_ratio = 0.45;
    static constexpr double face_confidence_threshold = 0.65;
    static constexpr double temporal_iou_threshold = 0.25;

    std::vector<cv::Rect> prevFaces;
    bool showEdgeWindow = true;
    bool edgeWindowCreated = false;

    /**
     * Process a single frame
     * @param frame Input BGR frame from video capture
     * @param grayFrame Grayscale/enhanced frame for detection
     * @param edges Edge map
     * @return Detected and smoothed face rectangles
     */
    std::vector<cv::Rect> processFrame(const cv::Mat& frame, const cv::Mat& grayFrame, const cv::Mat& edges);

    /**
     * Filter detections based on edge density
     * @param fusedDetections Detections from fusion
     * @param edges Edge map
     * @return Filtered face rectangles
     */
    std::vector<cv::Rect> filterByEdgeDensity(
        const std::vector<std::pair<cv::Rect, double>>& fusedDetections,
        const cv::Mat& edges
    );

    /**
     * Apply temporal smoothing to reduce jitter
     * @param faces Current frame detections
     * @return Temporally smoothed detections
     */
    std::vector<cv::Rect> applyTemporalSmoothing(const std::vector<cv::Rect>& faces);

    /**
     * Draw detections on frame
     * @param frame Frame to draw on
     * @param faces Detected faces
     * @param edges Edge map for overlay
     */
    void drawDetections(cv::Mat& frame, const std::vector<cv::Rect>& faces, const cv::Mat& edges);

    /**
     * Handle keyboard input
     * @return false if user wants to quit, true otherwise
     */
    bool handleInput();
};
