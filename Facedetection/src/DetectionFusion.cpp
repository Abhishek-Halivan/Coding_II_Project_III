#include "DetectionFusion.h"
#include <algorithm>
#include <cmath>

double DetectionFusion::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);

    const int interW = std::max(0, x2 - x1);
    const int interH = std::max(0, y2 - y1);
    const int interArea = interW * interH;
    const int unionArea = a.area() + b.area() - interArea;

    if (unionArea <= 0) {
        return 0.0;
    }
    return static_cast<double>(interArea) / static_cast<double>(unionArea);
}

cv::Rect DetectionFusion::smoothRect(const cv::Rect& prev, const cv::Rect& curr, double alpha) {
    const auto blend = [alpha](int p, int c) {
        return static_cast<int>(std::lround(alpha * p + (1.0 - alpha) * c));
    };

    return cv::Rect(
        blend(prev.x, curr.x),
        blend(prev.y, curr.y),
        blend(prev.width, curr.width),
        blend(prev.height, curr.height)
    );
}

std::vector<std::pair<cv::Rect, double>> DetectionFusion::fuseDetections(
    const std::vector<cv::Rect>& haarFaces,
    const std::vector<cv::Rect>& dnnFaces) {
    
    std::vector<std::pair<cv::Rect, double>> fusedFaces;
    fusedFaces.reserve(haarFaces.size() + dnnFaces.size());

    std::vector<bool> dnnUsed(dnnFaces.size(), false);

    for (const auto& haarFace : haarFaces) {
        double maxIou = 0.0;
        int bestDnnIdx = -1;

        for (int i = 0; i < static_cast<int>(dnnFaces.size()); ++i) {
            if (dnnUsed[i]) continue;

            const double iou = computeIoU(haarFace, dnnFaces[i]);
            if (iou > maxIou) {
                maxIou = iou;
                bestDnnIdx = i;
            }
        }

        if (bestDnnIdx >= 0 && maxIou > iou_threshold) {
            dnnUsed[bestDnnIdx] = true;
            const double confidence = fusion_base_confidence + 0.15 * maxIou;
            const cv::Rect merged(
                (haarFace.x + dnnFaces[bestDnnIdx].x) / 2,
                (haarFace.y + dnnFaces[bestDnnIdx].y) / 2,
                (haarFace.width + dnnFaces[bestDnnIdx].width) / 2,
                (haarFace.height + dnnFaces[bestDnnIdx].height) / 2
            );
            fusedFaces.emplace_back(merged, confidence);
        } else {
            fusedFaces.emplace_back(haarFace, haar_only_confidence);
        }
    }

    for (int i = 0; i < static_cast<int>(dnnFaces.size()); ++i) {
        if (!dnnUsed[i]) {
            fusedFaces.emplace_back(dnnFaces[i], dnn_only_confidence);
        }
    }

    return fusedFaces;
}
