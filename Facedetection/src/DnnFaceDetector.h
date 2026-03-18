#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

/**
 * DNN-based face detector using TensorFlow model
 */
class DnnFaceDetector {
public:
    DnnFaceDetector();

    /**
     * Automatically find and load DNN model
     * @return true if model found and loaded, false otherwise
     */
    bool loadAuto();

    /**
     * Load DNN model from explicit paths
     * @param modelPath Path to .pb model file
     * @param configPath Path to .pbtxt config file
     * @return true if successfully loaded, false otherwise
     */
    bool load(const std::string& modelPath, const std::string& configPath);

    /**
     * Detect faces in frame
     * @param frame Input frame (BGR)
     * @param confidenceThreshold Confidence threshold for detection
     * @return Vector of detected face rectangles
     */
    std::vector<cv::Rect> detect(const cv::Mat& frame, double confidenceThreshold = 0.5);

    bool isLoaded() const { return !net.empty(); }

private:
    cv::dnn::Net net;
    std::string findDnnModel();
    static constexpr int blob_size = 300;
    static constexpr double blob_scale = 1.0;
};
