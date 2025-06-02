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
#include <QRandomGenerator>

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

std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& img)
{
    thread_local dlib::rand rnd;
    std::vector<dlib::matrix<dlib::rgb_pixel>> crops;
    for (int i = 0; i < 10; ++i)
        crops.push_back(dlib::jitter_image(img, rnd));
    return crops;
}


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

std::vector<QRect> FaceIndexer::detectFaces(const QImage& image) {
    cv::Mat mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());
    cv::Mat matBGR;
    cv::cvtColor(mat, matBGR, cv::COLOR_RGBA2BGR);
    return detectFaces(matBGR);
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

std::vector<float> FaceIndexer::getJitteredEmbedding(const cv::Mat& image, const QRect& faceRect) {
    std::vector<float> descriptor;
    if (!impl) return descriptor;

    dlib::cv_image<dlib::bgr_pixel> cimg(image);
    dlib::rectangle faceBox(faceRect.x(), faceRect.y(),
                            faceRect.x() + faceRect.width(),
                            faceRect.y() + faceRect.height());

    auto shape = impl->sp(cimg, faceBox);
    dlib::matrix<dlib::rgb_pixel> face_chip;
    extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);

    auto jitters = jitter_image(face_chip);
    dlib::matrix<float, 0, 1> desc = dlib::mean(dlib::mat(impl->net(jitters)));

    descriptor.assign(desc.begin(), desc.end());
    return descriptor;
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

    // Use jittering for robust embedding
    std::vector<matrix<rgb_pixel>> jitters = jitter_image(face_chip);
    matrix<float, 0, 1> faceDesc = mean(mat(impl->net(jitters)));

    descriptor.assign(faceDesc.begin(), faceDesc.end());
    return descriptor;

}

void FaceIndexer::buildOrUpdateFaceLog(const QString& folderPath) {
    qDebug() << "Building face log for:" << folderPath;

    QString cleanPath = QDir::cleanPath(folderPath).toLower();
    if (cleanPath.contains("/.cache/") || cleanPath.contains("\\.cache\\")) {
        qDebug() << "Skipping facelog generation for cached folder:" << folderPath;
        return;
    }

    QString logFilePath = folderPath + "/.facelog.json";
    QString globalLogPath = "C:/PhotoExplorer/faces/global_facelog.json";
    QDir().mkpath(QFileInfo(globalLogPath).absolutePath());

    QJsonArray existing = loadFaceLog(logFilePath);
    QJsonArray globalLog = loadFaceLog(globalLogPath);
    QSet<QString> knownPaths;
    QSet<QString> stillUsedFaceIds;

    // Step 1: Prune deleted images
    QJsonArray prunedEntries;
    for (const auto& val : existing) {
        QJsonObject obj = val.toObject();
        QString imgPath = obj["image"].toString();
        if (QFile::exists(imgPath)) {
            prunedEntries.append(obj);
            knownPaths.insert(imgPath);
            stillUsedFaceIds.insert(obj["id"].toString());
        } else {
            qDebug() << "Removing deleted image from log:" << imgPath;
            if (obj.contains("thumb")) {
                QString thumb = obj["thumb"].toString();
                if (QFile::exists(thumb)) QFile::remove(thumb);
            }
        }
    }

    // Step 2: Scan current folder for new images
    QDirIterator it(folderPath, QStringList() << "*.jpg" << "*.png", QDir::Files);
    QString faceThumbDir = folderPath + "/.cache/faces";
    QDir().mkpath(faceThumbDir);

    while (it.hasNext()) {
        QString path = it.next();

        // Skip .cache or already processed
        QDir parent = QFileInfo(path).dir();
        bool skip = false;
        while (parent.cdUp()) {
            if (parent.dirName().toLower() == ".cache") {
                skip = true;
                break;
            }
        }
        if (skip || knownPaths.contains(path)) {
            qDebug() << "Skipping already-processed:" << path;
            continue;
        }

        cv::Mat img = cv::imread(path.toStdString());
        if (img.empty()) continue;

        QString logLine = QString("Image: %1 | Size: %2x%3").arg(path).arg(img.cols).arg(img.rows);

        auto rects = detectFaces(img);
        if (rects.empty()) {
            qDebug() << logLine << "| Status: no-face";
            continue;
        }

        int faceIndex = 0;
        for (const auto& r : rects) {
            cv::Rect roi(r.x(), r.y(), r.width(), r.height());
            roi = roi & cv::Rect(0, 0, img.cols, img.rows);

            if (roi.width < 40 || roi.height < 40) {
                qDebug() << logLine << QString("| Face #%1: skipped-small | ROI: %2x%3")
                .arg(faceIndex).arg(roi.width).arg(roi.height);
                faceIndex++;
                continue;
            }

            if (roi.width > img.cols * 0.9 || roi.height > img.rows * 0.9) {
                qDebug() << logLine << QString("| Face #%1: skipped-large | ROI: %2x%3")
                .arg(faceIndex).arg(roi.width).arg(roi.height);
                faceIndex++;
                continue;
            }

            auto embedding = getFaceEmbedding(img, r);
            if (embedding.empty()) {
                qDebug() << logLine << QString("| Face #%1: skipped-embedding-fail").arg(faceIndex);
                faceIndex++;
                continue;
            }

            QString thumbPath = faceThumbDir + "/" + QFileInfo(path).baseName()
                                + QString("_face%1.jpg").arg(faceIndex);

            cv::Mat faceCrop = img(roi), resizedFace;
            cv::resize(faceCrop, resizedFace, cv::Size(64, 64));
            cv::imwrite(thumbPath.toStdString(), faceCrop);

            QJsonObject entry;
            QString faceId = assignGlobalFaceId(embedding, globalLog);
            stillUsedFaceIds.insert(faceId);

            entry["id"] = faceId;
            entry["image"] = path;
            entry["thumb"] = thumbPath;

            QJsonArray vec;
            for (float val : embedding) vec.append(val);
            entry["embedding"] = vec;

            prunedEntries.append(entry);
            qDebug() << logLine << QString("| Face #%1: logged | ROI: %2x%3")
                                       .arg(faceIndex).arg(roi.width).arg(roi.height);

            emit faceLogged(faceId, thumbPath);
            faceIndex++;
        }
    }

    // Step 3: Prune global log of unused face IDs
    QJsonArray cleanedGlobal;
    for (const auto& val : globalLog) {
        QJsonObject obj = val.toObject();
        QString id = obj["id"].toString();
        if (stillUsedFaceIds.contains(id)) {
            cleanedGlobal.append(obj);
        } else {
            qDebug() << "Removing unused face ID:" << id;
        }
    }

    saveFaceLog(logFilePath, prunedEntries);
    saveFaceLog(globalLogPath, cleanedGlobal);

    qDebug() << "Saved facelog to:" << logFilePath << "entries:" << prunedEntries.size();
    qDebug() << "Saved global facelog to:" << globalLogPath << "entries:" << cleanedGlobal.size();
}


QSet<QString> FaceIndexer::collectUsedFaceIdsFromAllDrives() {
    QSet<QString> allUsedIds;
    QFileInfoList drives = QDir::drives();

    for (const QFileInfo& drive : drives) {
        QString root = drive.absoluteFilePath();

        QDirIterator it(root,
                        QStringList() << ".facelog.json",
                        QDir::Files,
                        QDirIterator::Subdirectories);

        while (it.hasNext()) {
            QString facelogPath = it.next();

            // Skip system/hidden folders
            QFileInfo info(facelogPath);
            if (info.absoluteFilePath().contains("/$") || info.isHidden())
                continue;

            QJsonArray log = loadFaceLog(facelogPath);
            for (const auto& val : log) {
                QJsonObject obj = val.toObject();
                QString id = obj["id"].toString();
                if (!id.isEmpty()) {
                    allUsedIds.insert(id);
                }
            }
        }
    }

    return allUsedIds;
}




void FaceIndexer::saveFaceLog(const QString& path, const QJsonArray& entries) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) return;
    QJsonDocument doc(entries);
    f.write(doc.toJson());
    f.close();
}

QJsonArray FaceIndexer::loadFaceLog(const QString& path) const{
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
        map[obj["id"].toString()] = obj.contains("thumb") ? obj["thumb"].toString() : obj["image"].toString();
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

        if (dist < 0.5f)
            results.append(obj["image"].toString());
    }
    return results;
}

QJsonArray FaceIndexer::getFaceLogRaw(const QString& folderPath) {
    return loadFaceLog(folderPath + "/.facelog.json");
}

QString FaceIndexer::findMatchingId(const std::vector<float>& newEmbedding, const QJsonArray& entries) {
    for (const auto& val : entries) {
        QJsonObject obj = val.toObject();
        QJsonArray embArray = obj["embedding"].toArray();
        if (embArray.size() != int(newEmbedding.size())) continue;

        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (int i = 0; i < embArray.size(); ++i) {
            float a = float(embArray[i].toDouble());
            float b = newEmbedding[i];
            dot += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }

        float cosine = dot / (std::sqrt(norm1) * std::sqrt(norm2));
        if (cosine > 0.85f)  // you can adjust threshold
            return obj["id"].toString();
    }
    return "";
}


QString FaceIndexer::assignGlobalFaceId(const std::vector<float>& emb, QJsonArray& globalLog) {
    for (const auto& val : globalLog) {
        QJsonObject obj = val.toObject();
        QJsonArray embArray = obj["embedding"].toArray();
        if (embArray.size() != int(emb.size())) continue;

        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (int i = 0; i < emb.size(); ++i) {
            float a = float(embArray[i].toDouble());
            float b = emb[i];
            dot += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }

        float cosine = dot / (std::sqrt(norm1) * std::sqrt(norm2));
        if (cosine > 0.85f)  // match found
            return obj["id"].toString();
    }

    // No match → create new ID
    QString newId = QString("face_%1").arg(globalLog.size());
    QJsonObject newEntry;
    newEntry["id"] = newId;

    QJsonArray vec;
    for (float val : emb) vec.append(val);
    newEntry["embedding"] = vec;
    globalLog.append(newEntry);

    return newId;
}

QJsonArray FaceIndexer::getGlobalLog() const {
    QString globalLogPath = "C:/PhotoExplorer/faces/global_facelog.json";
    return loadFaceLog(globalLogPath);
}

void FaceIndexer::saveFaceLogAtomic(const QString& path, const QJsonArray& entries) {
    QString tempPath = path + ".tmp";

    QFile f(tempPath);
    if (!f.open(QIODevice::WriteOnly)) {
        qWarning() << "❌ Could not open temp file for saving:" << tempPath;
        return;
    }

    QJsonDocument doc(entries);
    f.write(doc.toJson());
    f.close();

    QFile::remove(path);  // optional: ensure clean overwrite
    if (!QFile::rename(tempPath, path)) {
        qWarning() << "❌ Failed to rename temp file to target:" << path;
    }
}




