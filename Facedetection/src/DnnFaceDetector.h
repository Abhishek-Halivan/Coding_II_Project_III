#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

// Deep neural network-based face detector using OpenCV's DNN module with a TensorFlow ResNet model.
// More accurate than Haar cascades but requires DNN model files (.pb and .pbtxt).
// Combined with Haar detector for hybrid approach that balances speed and accuracy.
class DnnFaceDetector {
public:
    DnnFaceDetector();

    // Automatically search for and load DNN model files from standard OpenCV data paths.
    // Looks in vcpkg buildtrees and default OpenCV installation directories.
    // Returns true if model found and successfully loaded, false if not found.
    bool loadAuto();

    // Load DNN model from explicit file paths.
    // modelPath: path to .pb (protobuf) model file
    // configPath: path to .pbtxt (config) file
    // Returns true on success, false if files not found or model load fails.
    bool load(const std::string& modelPath, const std::string& configPath);

    // Detect faces in the input frame using the loaded DNN model.
    // Converts frame to 300x300 blob for inference (standard ResNet face detector input size).
    // confidenceThreshold: filters detections with confidence below this value (typically 0.5-0.6).
    // Returns vector of detected face rectangles scaled back to original frame size.
    std::vector<cv::Rect> detect(const cv::Mat& frame, double confidenceThreshold = 0.5);

    // Check if DNN model has been loaded and is ready for inference.
    bool isLoaded() const { return !net.empty(); }

private:
    cv::dnn::Net net;
    std::string findDnnModel();
    static constexpr int blob_size = 300;
    static constexpr double blob_scale = 1.0;
};
