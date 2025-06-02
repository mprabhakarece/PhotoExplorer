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
    QStringList filterByFace(const QString& folderPath, const std::vector<float>& targetEmbedding);

private:
    QString predictorPath;
    QString recognitionModelPath;

    void loadModels();
    std::vector<QRect> detectFaces(const cv::Mat& image);
    std::vector<float> getFaceEmbedding(const cv::Mat& image, const QRect& faceRect);

    void saveFaceLog(const QString& path, const QJsonArray& entries);
    QJsonArray loadFaceLog(const QString& path);

    class Impl;
    Impl* impl;
};
