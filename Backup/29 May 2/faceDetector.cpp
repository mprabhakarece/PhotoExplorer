#include "faceDetector.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include <QDebug>
#include <QImage>
#include <QRect>

constexpr int JITTER_COUNT = 10;

using namespace dlib;

// Dlib model definition (ResNet-based)
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1, relu<BN<con<N,3,3,stride,stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares       = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down  = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                                                  alevel0<
                                                      alevel1<
                                                          alevel2<
                                                              alevel3<
                                                                  alevel4<
                                                                      max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                                                                                           input_rgb_image_sized<150>
                                                                                                           >>>>>>>>>>>>;

class FaceDetector::Impl {
public:
    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;

    Impl() {
        detector = get_frontal_face_detector();
        deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp;
        deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;
    }
};

// Constructor and destructor
FaceDetector::FaceDetector() {
    try {
        impl = new Impl();
    } catch (const std::exception& e) {
        qDebug() << "FaceDetector failed to initialize:" << e.what();
        impl = nullptr;
    }
}

FaceDetector::~FaceDetector() {
    delete impl;
}

// Detect faces from QImage
std::vector<QRect> FaceDetector::detectFaces(const QImage& image) {
    cv::Mat mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());
    cv::Mat matBGR;
    cv::cvtColor(mat, matBGR, cv::COLOR_RGBA2BGR);
    return detectFaces(matBGR);
}

// Detect faces from cv::Mat
std::vector<QRect> FaceDetector::detectFaces(const cv::Mat& image) {
    std::vector<QRect> faces;
    if (!impl) return faces;

    cv_image<bgr_pixel> dlibImg(image);
    std::vector<rectangle> dets = impl->detector(dlibImg);

    for (const auto& r : dets) {
        faces.push_back(QRect(r.left(), r.top(), r.width(), r.height()));
    }
    return faces;
}

// Compute 128D face embedding
std::vector<float> FaceDetector::getFaceEmbedding(const cv::Mat& image, const QRect& faceRect) {
    std::vector<float> descriptor;
    if (!impl) return descriptor;

    cv_image<bgr_pixel> cimg(image);
    rectangle faceBox(faceRect.x(), faceRect.y(), faceRect.x() + faceRect.width(), faceRect.y() + faceRect.height());

    full_object_detection shape = impl->sp(cimg, faceBox);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);

    matrix<float, 0, 1> faceDesc = impl->net(face_chip);
    descriptor.assign(faceDesc.begin(), faceDesc.end());

    return descriptor;
}

// Compare two cropped face images
bool FaceDetector::isMatchingFace(const cv::Mat& face1, const cv::Mat& face2, float threshold) {
    auto d1 = getFaceEmbedding(face1, QRect(0, 0, face1.cols, face1.rows));
    auto d2 = getFaceEmbedding(face2, QRect(0, 0, face2.cols, face2.rows));

    float dot = std::inner_product(d1.begin(), d1.end(), d2.begin(), 0.0f);
    float norm1 = std::sqrt(std::inner_product(d1.begin(), d1.end(), d1.begin(), 0.0f));
    float norm2 = std::sqrt(std::inner_product(d2.begin(), d2.end(), d2.begin(), 0.0f));

    return (dot / (norm1 * norm2)) > (1.0f - threshold);
}

// Generate 10 jittered variations
std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img) {
    thread_local dlib::rand rnd;
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops;
    for (int i = 0; i < JITTER_COUNT; ++i)
        crops.push_back(dlib::jitter_image(img, rnd));
    return crops;
}

// Robust embedding with jittering
std::vector<float> FaceDetector::getJitteredEmbedding(const cv::Mat& image, const QRect& faceRect) {
    std::vector<float> descriptor;
    if (!impl) return descriptor;

    dlib::cv_image<dlib::bgr_pixel> cimg(image);
    dlib::rectangle faceBox(faceRect.x(), faceRect.y(), faceRect.x() + faceRect.width(), faceRect.y() + faceRect.height());

    auto shape = impl->sp(cimg, faceBox);
    dlib::matrix<dlib::rgb_pixel> face_chip;
    extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);

    auto jitters = jitter_image(face_chip);
    dlib::matrix<float, 0, 1> desc = dlib::mean(mat(impl->net(jitters)));

    descriptor.assign(desc.begin(), desc.end());
    return descriptor;
}

dlib::full_object_detection FaceDetector::getLandmarks(const dlib::cv_image<dlib::bgr_pixel>& img, const dlib::rectangle& faceRect) {
    return impl->sp(img, faceRect);
}

