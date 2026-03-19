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

    try {
        recognizerNet = cv::dnn::readNetFromTorch("openface.nn4.small2.v1.t7");
    } catch (...) {
        std::cerr << "Could not load openface.nn4.small2.v1.t7. Face recording disabled.\n";
    }

    std::cout << "Press 'q' or ESC to quit.\n";
    std::cout << "Press 'e' to toggle edge preview window.\n";
    std::cout << "Press 'r' to record your face to exclude it from blurring.\n";

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
    if (requestRecordFace && !faces.empty()) {
        cv::Rect largestFace = faces[0];
        for (const auto& f : faces) {
            if (f.area() > largestFace.area()) largestFace = f;
        }
        myFaceEmbedding = getFaceEmbedding(frame, largestFace);
        if (!myFaceEmbedding.empty()) {
            isMyFaceRecorded = true;
            std::cout << "Face recorded successfully! Your face will no longer be blurred.\n";
        }
        requestRecordFace = false;
    }

    for (const auto& face : faces) {
        // Ensure the ROI is strictly within the frame boundaries to prevent crashes
        cv::Rect safeFace = face & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safeFace.area() <= 0) continue;

        bool isMe = false;
        if (isMyFaceRecorded) {
            cv::Mat currentEmbedding = getFaceEmbedding(frame, safeFace);
            double distance = compareFaces(myFaceEmbedding, currentEmbedding);
            // OpenFace typical threshold for the same person is an L2 distance < 0.8
            if (distance < 0.8) {
                isMe = true;
            }
        }

        if (isMe) {
            // Draw a subtle green border for the recorded user instead of blurring
            cv::rectangle(frame, safeFace, cv::Scalar(0, 255, 0), 2);
        } else {
            // Extract the Region of Interest (ROI)
            cv::Mat roi = frame(safeFace);

            // Calculate a strong blur kernel size based on face size (must be an odd number)
            int kernelSize = safeFace.width / 5;
            if (kernelSize % 2 == 0) kernelSize++; // Make sure it is odd
            if (kernelSize < 3) kernelSize = 3;    // Minimum viable kernel size for GaussianBlur

            // Apply OpenCV Gaussian Blur directly to the cropped Region of Interest
            cv::GaussianBlur(roi, roi, cv::Size(kernelSize, kernelSize), 0);
        }
    }
}

cv::Mat FaceDetectionApp::getFaceEmbedding(const cv::Mat& frame, const cv::Rect& faceRect) {
    if (recognizerNet.empty()) return cv::Mat();

    cv::Rect safeFace = faceRect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeFace.area() <= 0) return cv::Mat();

    cv::Mat faceCropped = frame(safeFace);
    // OpenFace expects 96x96 RGB image, scaling pixels to 0-1
    cv::Mat blob = cv::dnn::blobFromImage(faceCropped, 1.0/255.0, cv::Size(96, 96), cv::Scalar(0,0,0), true, false);
    recognizerNet.setInput(blob);
    return recognizerNet.forward().clone();
}

double FaceDetectionApp::compareFaces(const cv::Mat& emb1, const cv::Mat& emb2) {
    if (emb1.empty() || emb2.empty()) return 1.0;
    // Euclidian distance between the two 128-d vectors
    return cv::norm(emb1, emb2, cv::NORM_L2);
}

bool FaceDetectionApp::handleInput() {
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
        return false;
    }
    if (key == 'e' || key == 'E') {
        showEdgeWindow = !showEdgeWindow;
    }
    if (key == 'r' || key == 'R') {
        requestRecordFace = true;
    }
    return true;
}
