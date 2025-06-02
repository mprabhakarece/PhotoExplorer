#include "faceindexer.h"

#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QDebug>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <QDirIterator>
#include <QJsonArray>
#include <QJsonObject>
#include <QMap>


using namespace dlib;

// Use the full model definition from Dlib example
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

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
                                                                      max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
                                                                                                           input_rgb_image_sized<150>
                                                                                                           >>>>>>>>>>>>;


// === Internal Implementation ===
class FaceIndexer::Impl {
public:
    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;

    Impl(const QString& predictorPath, const QString& modelPath) {
        detector = get_frontal_face_detector();
        deserialize(predictorPath.toStdString()) >> sp;
        deserialize(modelPath.toStdString()) >> net;
    }
};

// === FaceIndexer API ===

FaceIndexer::FaceIndexer(const QString& modelPath)
{
    predictorPath = modelPath + "/shape_predictor_68_face_landmarks.dat";
    recognitionModelPath = modelPath + "/dlib_face_recognition_resnet_model_v1.dat";
    loadModels();
}

void FaceIndexer::loadModels() {
    impl = new Impl(predictorPath, recognitionModelPath);
}

std::vector<QRect> FaceIndexer::detectFaces(const cv::Mat& image) {
    std::vector<QRect> faces;
    if (!impl) return faces;

    dlib::cv_image<dlib::bgr_pixel> dlibImg(image);
    auto dets = impl->detector(dlibImg);
    for (auto& r : dets) {
        faces.push_back(QRect(r.left(), r.top(), r.width(), r.height()));
    }
    return faces;
}

std::vector<float> FaceIndexer::getFaceEmbedding(const cv::Mat& image, const QRect& faceRect) {
    std::vector<float> descriptor;
    if (!impl) return descriptor;

    dlib::cv_image<dlib::bgr_pixel> cimg(image);
    rectangle faceBox(faceRect.x(), faceRect.y(),
        faceRect.x() + faceRect.width(),
        faceRect.y() + faceRect.height());
    auto shape = impl->sp(cimg, faceBox);

    matrix<rgb_pixel> face_chip;
    extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
    matrix<float, 0, 1> faceDesc = impl->net(face_chip);

    descriptor.assign(faceDesc.begin(), faceDesc.end());
    return descriptor;
}

void FaceIndexer::buildOrUpdateFaceLog(const QString& folderPath) {
    QString logFilePath = folderPath + "/.facelog.json";
    QJsonArray existing = loadFaceLog(logFilePath);
    QSet<QString> knownPaths;
    for (auto val : existing) {
        QString path = val.toObject()["image"].toString();
        knownPaths.insert(path);
    }

    QJsonArray newEntries = existing;
    QDirIterator it(folderPath, QStringList() << "*.jpg" << "*.png", QDir::Files);
    while (it.hasNext()) {
        QString path = it.next();
        if (knownPaths.contains(path)) continue;

        cv::Mat img = cv::imread(path.toStdString());
        if (img.empty()) continue;

        auto rects = detectFaces(img);
        if (rects.empty()) continue;

        auto embedding = getFaceEmbedding(img, rects[0]);
        if (embedding.empty()) continue;

        QJsonObject entry;
        entry["id"] = QString::number(qHash(path));
        entry["image"] = path;

        QJsonArray vec;
        for (float val : embedding) vec.append(val);
        entry["embedding"] = vec;

        newEntries.append(entry);
        qDebug() << "✔ Logged face from" << path;
    }

    saveFaceLog(logFilePath, newEntries);
}

void FaceIndexer::saveFaceLog(const QString& path, const QJsonArray& entries) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) return;
    QJsonDocument doc(entries);
    f.write(doc.toJson());
    f.close();
}

QJsonArray FaceIndexer::loadFaceLog(const QString& path) {
    QFile f(path);
    if (!f.exists() || !f.open(QIODevice::ReadOnly)) return {};
    QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    f.close();
    return doc.array();
}

QMap<QString, QString> FaceIndexer::getKnownFaces(const QString& folderPath) {
    QMap<QString, QString> map;
    QJsonArray entries = loadFaceLog(folderPath + "/.facelog.json");
    for (const auto& val : entries) {
        QJsonObject obj = val.toObject();
        map[obj["id"].toString()] = obj["image"].toString();
    }
    return map;
}

QStringList FaceIndexer::filterByFace(const QString& folderPath, const std::vector<float>& targetEmbedding) {
    QStringList results;
    QJsonArray entries = loadFaceLog(folderPath + "/.facelog.json");
    for (const auto& val : entries) {
        QJsonObject obj = val.toObject();
        QJsonArray embArray = obj["embedding"].toArray();

        if (embArray.size() != int(targetEmbedding.size())) continue;

        float dist = 0.0f;
        for (int i = 0; i < embArray.size(); ++i) {
            float diff = float(embArray[i].toDouble()) - targetEmbedding[i];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);

        if (dist < 0.6f)
            results.append(obj["image"].toString());
    }
    return results;
}

