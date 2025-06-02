#pragma once

#include <vector>
#include <QString>
#include <QPixmap>
#include <dlib/image_processing.h>
#include <numeric>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

struct FaceStats {
    std::vector<float> embedding;
    double symmetry;
    double focus;
    QPixmap thumb;
    QString imagePath;
    QRect faceRect;
    int count = 1;
};

inline float l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    float normA = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
    float normB = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));
    return (normA > 0 && normB > 0) ? (dot / (normA * normB)) : 0.0f;
}

inline bool eyesAreOpen(const dlib::full_object_detection& shape) {
    auto eyeOpenness = [&](int top1, int top2, int bottom1, int bottom2) {
        return (shape.part(bottom1).y() + shape.part(bottom2).y()) -
               (shape.part(top1).y() + shape.part(top2).y());
    };
    double leftEye = eyeOpenness(37, 38, 41, 40);
    double rightEye = eyeOpenness(43, 44, 47, 46);
    double eyeOpenScore = (leftEye + rightEye) / 2.0;
    return eyeOpenScore > 4.0;
}

inline double getSymmetryScore(const dlib::full_object_detection& shape) {
    double eyeCenter = (shape.part(36).x() + shape.part(45).x()) / 2.0;
    double noseX = shape.part(30).x();
    return std::abs(eyeCenter - noseX);
}

inline double getFocusScore(const cv::Mat& faceMat) {
    cv::Mat gray, lap;
    cv::cvtColor(faceMat, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

inline bool isSimilarFace(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
    return l2Distance(a, b) < threshold;
}

inline bool isBetterMatch(const std::vector<float>& embedding,
                          FaceStats& existing,
                          const QPixmap& thumb,
                          const QString& path,
                          const dlib::full_object_detection& shape,
                          double focus,
                          double symmetry,
                          float matchDist = 0.5f,
                          double goodFocusThreshold = 100.0,
                          double focusTolerance = 25.0) {
    if (!isSimilarFace(embedding, existing.embedding, matchDist)) return false;

    if (!eyesAreOpen(shape)) return false;

    double prevFocus = existing.focus;
    bool focusGood = focus >= goodFocusThreshold;
    bool focusAcceptable = focus >= (prevFocus - focusTolerance);

    if (symmetry < existing.symmetry && (focusGood || focusAcceptable)) {
        existing.embedding = embedding;
        existing.symmetry = symmetry;
        existing.focus = focus;
        existing.thumb = thumb;
        existing.imagePath = path;
        return true;
    }
    return false;
}

inline std::vector<float> normalizeEmbedding(const std::vector<float>& emb) {
    float norm = std::sqrt(std::inner_product(emb.begin(), emb.end(), emb.begin(), 0.0f));
    std::vector<float> result(emb.size());
    if (norm > 0.0f) {
        std::transform(emb.begin(), emb.end(), result.begin(), [=](float v) { return v / norm; });
    }
    return result;
}
