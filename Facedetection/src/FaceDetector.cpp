#include "FaceDetector.h"
#include <opencv2/imgproc.hpp>
#include <cstdlib>
#include <iostream>

FaceDetector::FaceDetector() = default;

bool FaceDetector::load(const std::string& cascadePath) {
    std::string path = cascadePath;

    if (path.empty()) {
        const char* envPath = std::getenv("OPENCV_FACE_CASCADE");
        if (envPath != nullptr) {
            path = envPath;
        } else {
            path = "haarcascade_frontalface_default.xml";
        }
    }

    if (!cascade.load(path)) {
        std::cerr << "Could not load cascade file: " << path << "\n";
        std::cerr << "Usage: face_detection <path-to-haarcascade_frontalface_default.xml>\n";
        std::cerr << "Or set OPENCV_FACE_CASCADE environment variable.\n";
        return false;
    }

    std::cout << "Successfully loaded Haar cascade from: " << path << "\n";
    return true;
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(
        frame,
        faces,
        scale_factor,
        min_neighbors,
        cv::CASCADE_SCALE_IMAGE,
        cv::Size(min_face_size, min_face_size)
    );
    return faces;
}

std::vector<cv::Rect> FaceDetector::detectMultiScale(const cv::Mat& grayFrame, double scale) {
    if (scale >= 1.0) {
        return detect(grayFrame);
    }

    cv::Mat small;
    cv::resize(grayFrame, small, cv::Size(), scale, scale, cv::INTER_LINEAR);

    std::vector<cv::Rect> facesSmall;
    cascade.detectMultiScale(
        small,
        facesSmall,
        scale_factor,
        min_neighbors,
        cv::CASCADE_SCALE_IMAGE,
        cv::Size(min_face_size, min_face_size)
    );

    std::vector<cv::Rect> faces;
    faces.reserve(facesSmall.size());

    for (const auto& f : facesSmall) {
        cv::Rect scaledFace(
            static_cast<int>(std::lround(f.x / scale)),
            static_cast<int>(std::lround(f.y / scale)),
            static_cast<int>(std::lround(f.width / scale)),
            static_cast<int>(std::lround(f.height / scale))
        );
        scaledFace &= cv::Rect(0, 0, grayFrame.cols, grayFrame.rows);
        if (scaledFace.width > 0 && scaledFace.height > 0) {
            faces.push_back(scaledFace);
        }
    }

    return faces;
}
