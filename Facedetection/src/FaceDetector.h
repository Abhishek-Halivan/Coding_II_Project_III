#pragma once

#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

/**
 * Haar cascade-based face detector
 */
class FaceDetector {
public:
    FaceDetector();

    /**
     * Load Haar cascade classifier from file or environment variable
     * @return true if successfully loaded, false otherwise
     */
    bool load(const std::string& cascadePath = "");

    /**
     * Detect faces in a frame
     * @param frame Input frame (BGR or grayscale)
     * @return Vector of detected face rectangles
     */
    std::vector<cv::Rect> detect(const cv::Mat& frame);

    /**
     * Detect faces in a grayscale frame
     * @param grayFrame Input grayscale frame
     * @param scale detection scale (0.5 means detect on 50% scaled frame)
     * @return Vector of detected face rectangles (scaled back to original)
     */
    std::vector<cv::Rect> detectMultiScale(const cv::Mat& grayFrame, double scale = 1.0);

    bool isLoaded() const { return cascade.empty() == false; }

private:
    cv::CascadeClassifier cascade;
    static constexpr double scale_factor = 1.05;
    static constexpr int min_neighbors = 7;
    static constexpr int min_face_size = 28;
};
