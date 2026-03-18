#include "FaceDetectionApp.h"
#include "DetectionFusion.h"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

FaceDetectionApp::FaceDetectionApp()
    : showEdgeWindow(true) {
}

bool FaceDetectionApp::initialize(const std::string& cascadePath) {
    if (!haarDetector.load(cascadePath)) {
        return false;
    }

    dnnDetector.loadAuto();

    std::cout << "Press 'q' or ESC to quit.\n";
    std::cout << "Press 'e' to toggle edge preview window.\n";

    return true;
}

void FaceDetectionApp::run() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open default camera.\n";
        return;
    }

    cv::Mat frame;
    cv::Mat gray;
    cv::Mat enhancedGray;
    cv::Mat edges;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Received empty frame from camera.\n";
            break;
        }

        // Convert to grayscale and enhance
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        enhancedGray = edgeDetector.enhance(gray);

        // Detect edges
        edges = edgeDetector.detectEdges(enhancedGray);

        // Process frame to get detected faces
        std::vector<cv::Rect> faces = processFrame(frame, enhancedGray, edges);

        // Draw detections
        drawDetections(frame, faces, edges);

        // Display results
        cv::imshow("Face Detection", frame);
        if (showEdgeWindow) {
            cv::imshow("Face Edges", edges);
            edgeWindowCreated = true;
        } else if (edgeWindowCreated) {
            cv::destroyWindow("Face Edges");
            edgeWindowCreated = false;
        }

        // Handle input
        if (!handleInput()) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

std::vector<cv::Rect> FaceDetectionApp::processFrame(const cv::Mat& frame, const cv::Mat& grayFrame, const cv::Mat& edges) {
    // Detect with both methods
    std::vector<cv::Rect> haarFaces = haarDetector.detectMultiScale(
        grayFrame,
        detection_scale
    );

    std::vector<cv::Rect> dnnFaces = dnnDetector.detect(frame, dnn_confidence_threshold);

    // Fuse detections
    auto fusedDetections = DetectionFusion::fuseDetections(haarFaces, dnnFaces);

    // Filter by edge density
    std::vector<cv::Rect> filteredFaces = filterByEdgeDensity(fusedDetections, edges);

    // Apply temporal smoothing
    std::vector<cv::Rect> stableFaces = applyTemporalSmoothing(filteredFaces);

    prevFaces = stableFaces;

    return stableFaces;
}

std::vector<cv::Rect> FaceDetectionApp::filterByEdgeDensity(
    const std::vector<std::pair<cv::Rect, double>>& fusedDetections,
    const cv::Mat& edges) {
    
    std::vector<cv::Rect> faces;
    faces.reserve(fusedDetections.size());

    for (const auto& [faceRect, confidence] : fusedDetections) {
        const double edgeRatio = edgeDetector.getEdgeDensityRatio(edges, faceRect);

        if (edgeRatio >= min_edge_ratio && edgeRatio <= max_edge_ratio && confidence >= face_confidence_threshold) {
            faces.push_back(faceRect);
        }
    }

    return faces;
}

std::vector<cv::Rect> FaceDetectionApp::applyTemporalSmoothing(const std::vector<cv::Rect>& faces) {
    std::vector<cv::Rect> stableFaces;
    stableFaces.reserve(faces.size());

    for (const auto& face : faces) {
        double bestIou = 0.0;
        int bestPrevIdx = -1;

        for (int i = 0; i < static_cast<int>(prevFaces.size()); ++i) {
            const double overlap = DetectionFusion::computeIoU(face, prevFaces[i]);
            if (overlap > bestIou) {
                bestIou = overlap;
                bestPrevIdx = i;
            }
        }

        if (bestPrevIdx >= 0 && bestIou > temporal_iou_threshold) {
            stableFaces.push_back(DetectionFusion::smoothRect(prevFaces[bestPrevIdx], face, temporal_alpha));
        } else {
            stableFaces.push_back(face);
        }
    }

    return stableFaces;
}

void FaceDetectionApp::drawDetections(cv::Mat& frame, const std::vector<cv::Rect>& faces, const cv::Mat& edges) {
    for (const auto& face : faces) {
        // Draw edge overlay
        frame(face).setTo(cv::Scalar(0, 255, 255), edges(face));
        // Draw rectangle border
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
    }
}

bool FaceDetectionApp::handleInput() {
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
        return false;
    }
    if (key == 'e' || key == 'E') {
        showEdgeWindow = !showEdgeWindow;
    }
    return true;
}
