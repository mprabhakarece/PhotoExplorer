#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <QImage>
#include <QRect>
#include <vector>
#include <opencv2/core.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

using namespace dlib;

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    // Detect faces from QImage (Qt)
    std::vector<QRect> detectFaces(const QImage& image);

    // Detect faces from cv::Mat (OpenCV)
    std::vector<QRect> detectFaces(const cv::Mat& image);

    // Get 128D embedding for a single face (non-jittered)
    std::vector<float> getFaceEmbedding(const cv::Mat& image, const QRect& faceRect);

    // Get 128D embedding using 10-jittered samples (more robust)
    std::vector<float> getJitteredEmbedding(const cv::Mat& image, const QRect& faceRect);

    // Compare two aligned face crops (0,0,width,height) using cosine similarity
    bool isMatchingFace(const cv::Mat& face1, const cv::Mat& face2, float threshold = 0.6f);

    dlib::full_object_detection getLandmarks(const dlib::cv_image<dlib::bgr_pixel>& img, const dlib::rectangle& faceRect);

private:
    class Impl;
    Impl* impl;
};

#endif // FACEDETECTOR_H
