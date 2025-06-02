#pragma once

#include <QString>
#include <QRect>
#include <QMap>
#include <QStringList>
#include <QJsonArray>
#include <opencv2/core.hpp>

class FaceIndexer {
public:
    explicit FaceIndexer(const QString& modelPath);  // Pass folder containing .dat files

    void buildOrUpdateFaceLog(const QString& folderPath);
    QMap<QString, QString> getKnownFaces(const QString& folderPath); // { id : sample_image_path }

    QJsonArray getFaceLogRaw(const QString& folderPath);
    QStringList filterByFace(const QString& folderPath, const std::vector<float>& targetEmbedding);
    QString findMatchingId(const std::vector<float>& newEmbedding, const QJsonArray& entries);
    QString assignGlobalFaceId(const std::vector<float>& emb, QJsonArray& globalLog);
    void saveFaceLog(const QString& path, const QJsonArray& entries);
    QSet<QString> collectUsedFaceIdsFromAllDrives();

private:
    QString predictorPath;
    QString recognitionModelPath;

    void loadModels();
    QJsonArray loadFaceLog(const QString& path);
    class Impl;
    Impl* impl;

public:
    std::vector<QRect> detectFaces(const cv::Mat& image);
    std::vector<float> getFaceEmbedding(const cv::Mat& image, const QRect& faceRect);
};
