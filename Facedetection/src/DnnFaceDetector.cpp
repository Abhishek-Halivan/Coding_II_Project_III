#include "DnnFaceDetector.h"
#include <filesystem>
#include <iostream>

DnnFaceDetector::DnnFaceDetector() = default;

std::string DnnFaceDetector::findDnnModel() {
    const std::vector<std::string> searchPaths = {
        "opencv_face_detector_uint8.pb",
        "C:\\vcpkg\\buildtrees\\opencv4\\src\\4.12.0-d26ced7cc8.clean\\samples\\dnn\\face_detector\\opencv_face_detector_uint8.pb",
        "C:\\opencv\\samples\\dnn\\face_detector\\opencv_face_detector_uint8.pb"
    };

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return "";
}

bool DnnFaceDetector::loadAuto() {
    std::string modelPath = findDnnModel();
    
    if (modelPath.empty()) {
        std::cout << "DNN face detector model not found. Using Haar cascade only.\n";
        return false;
    }

    const size_t lastDot = modelPath.find_last_of('.');
    std::string configPath = modelPath.substr(0, lastDot) + ".pbtxt";
    
    if (!std::filesystem::exists(configPath)) {
        std::cout << "DNN config file not found. Using Haar cascade only.\n";
        return false;
    }

    return load(modelPath, configPath);
}

bool DnnFaceDetector::load(const std::string& modelPath, const std::string& configPath) {
    try {
        net = cv::dnn::readNetFromTensorflow(modelPath, configPath);
        std::cout << "Loaded DNN face detector from: " << modelPath << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load DNN model: " << e.what() << "\n";
        return false;
    }
}

std::vector<cv::Rect> DnnFaceDetector::detect(const cv::Mat& frame, double confidenceThreshold) {
    std::vector<cv::Rect> faces;
    
    if (net.empty()) {
        return faces;
    }

    cv::Mat blob = cv::dnn::blobFromImage(frame, blob_scale, cv::Size(blob_size, blob_size),
                                           cv::Scalar(104, 177, 123), false, false);
    net.setInput(blob);
    cv::Mat detections = net.forward();

    const int rows = detections.size[2];
    const float* data = detections.ptr<float>();

    for (int i = 0; i < rows; ++i) {
        const float confidence = data[i * 7 + 2];

        if (confidence >= confidenceThreshold) {
            const int x1 = static_cast<int>(data[i * 7 + 3] * frame.cols);
            const int y1 = static_cast<int>(data[i * 7 + 4] * frame.rows);
            const int x2 = static_cast<int>(data[i * 7 + 5] * frame.cols);
            const int y2 = static_cast<int>(data[i * 7 + 6] * frame.rows);

            const int w = std::max(1, x2 - x1);
            const int h = std::max(1, y2 - y1);

            faces.emplace_back(x1, y1, w, h);
        }
    }
    return faces;
}
