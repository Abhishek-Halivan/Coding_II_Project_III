#pragma once

#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

// Classical Haar cascade-based face detector using OpenCV's pre-trained cascade classifier.
// Fast and lightweight, works well with clear, frontal face angles.
// Combines with DNN detector in the main app for improved accuracy.
class FaceDetector {
public:
    FaceDetector();

    // Load the Haar cascade classifier from file path.
    // If cascadePath is empty, checks OPENCV_FACE_CASCADE environment variable.
    // Returns true if cascade loaded successfully, false if file not found or load fails.
    bool load(const std::string& cascadePath = "");

    // Detect faces in input frame using single-scale detection.
    // Automatically converts to grayscale if input is color (BGR).
    // Returns vector of bounding rectangles for detected faces.
    std::vector<cv::Rect> detect(const cv::Mat& frame);

    // Perform multi-scale face detection on grayscale input for more robust results.
    // scale parameter: downsampling factor (0.5 = detect on half-sized frame for stability).
    // Detections on scaled frame are automatically scaled back to original image coordinates.
    // Returns vector of detected face rectangles in original image space.
    std::vector<cv::Rect> detectMultiScale(const cv::Mat& grayFrame, double scale = 1.0);

    // Check if cascade classifier has been loaded successfully.
    bool isLoaded() const { return cascade.empty() == false; }

private:
    cv::CascadeClassifier cascade;
    static constexpr double scale_factor = 1.05;
    static constexpr int min_neighbors = 7;
    static constexpr int min_face_size = 28;
};
