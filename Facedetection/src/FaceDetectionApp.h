#pragma once

#include "FaceDetector.h"
#include "DnnFaceDetector.h"
#include "EdgeDetector.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

// Main application class that orchestrates the complete face detection pipeline.
// Manages initialization, video capture, detection fusion, temporal smoothing, and real-time visualization.
// Coordinates between Haar cascade detector, DNN detector, edge analysis, and frame rendering.
class FaceDetectionApp {
public:
    FaceDetectionApp();

    // Initialize the application: load cascade files, set up detectors, and configure video capture.
    // If cascadePath is empty, checks the OPENCV_FACE_CASCADE environment variable.
    // Returns true on success, false if cascade file not found or detectors fail to initialize.
    bool initialize(const std::string& cascadePath = "");

    // Main event loop: continuously capture frames from webcam, run detections, apply temporal smoothing,
    // and render results. Processes keyboard input (q/Esc to quit, e to toggle edge preview).
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

    // Process a single video frame through the complete detection pipeline:
    // 1. Run both Haar and DNN detectors on enhanced grayscale frame
    // 2. Fuse detections using IoU-based confidence scoring
    // 3. Filter fusion results using edge density analysis
    // 4. Apply temporal smoothing to reduce jitter
    // Returns final list of face bounding boxes with improved stability.
    std::vector<cv::Rect> processFrame(const cv::Mat& frame, const cv::Mat& grayFrame, const cv::Mat& edges);

    // Filter detections by analyzing edge density within bounding boxes.
    // Regions with too few edges (below min_edge_ratio) or too many edges (above max_edge_ratio)
    // are rejected as non-face detections. Applies stricter edge constraints to low-confidence detections.
    // Returns filtered rectangles that pass edge validation.
    std::vector<cv::Rect> filterByEdgeDensity(
        const std::vector<std::pair<cv::Rect, double>>& fusedDetections,
        const cv::Mat& edges
    );

    // Smooth detection positions across frames to reduce flickering and jitter.
    // Matches current detections to previous frame faces using spatial proximity.
    // Applies weighted averaging (temporal_alpha controls weight of history vs. new detection).
    // Unmatched faces are added directly; unmatched previous faces fade out.
    std::vector<cv::Rect> applyTemporalSmoothing(const std::vector<cv::Rect>& faces);

    // Render detected faces onto the output frame with visual feedback.
    // Draws green bounding boxes around faces and overlays detected edges in cyan.
    // Updates frame in-place; used for real-time display to user.
    void drawDetections(cv::Mat& frame, const std::vector<cv::Rect>& faces, const cv::Mat& edges);

    // Process keyboard input from OpenCV window.
    // Returns false if user pressed q or Esc (quit signal), true otherwise.
    // 'e' key toggles edge preview window visibility.
    // 'r' key records the largest face on screen to bypass blurring.
    bool handleInput();

    // Face recognition components for selective blurring
    cv::dnn::Net recognizerNet;
    cv::Mat myFaceEmbedding;
    bool isMyFaceRecorded = false;
    bool requestRecordFace = false;

    // Generates a 128-d embedding from a face crop using OpenFace DNN
    cv::Mat getFaceEmbedding(const cv::Mat& frame, const cv::Rect& faceRect);
    
    // Calculates L2 distance between two face embeddings
    double compareFaces(const cv::Mat& emb1, const cv::Mat& emb2);
};
