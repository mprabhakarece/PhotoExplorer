#pragma once

#include <QString>
#include <QRect>
#include <QMap>
#include <QStringList>
#include <QJsonArray>
#include <QSet>
#include <QObject>
#include <opencv2/core.hpp>
#include <QImage>

class FaceIndexer : public QObject {
    Q_OBJECT
public:
    explicit FaceIndexer(const QString& modelPath);  // Path to folder containing .dat models

    // === Face Log Management ===
    void buildOrUpdateFaceLog(const QString& folderPath);
    QMap<QString, QString> getKnownFaces(const QString& folderPath);  // { faceId : thumbnail/image path }
    QJsonArray getFaceLogRaw(const QString& folderPath);
    QStringList filterByFace(const QString& folderPath, const std::vector<float>& targetEmbedding);
    QString findMatchingId(const std::vector<float>& newEmbedding, const QJsonArray& entries);
    QString assignGlobalFaceId(const std::vector<float>& emb, QJsonArray& globalLog);
    void saveFaceLog(const QString& path, const QJsonArray& entries);
    void saveFaceLogAtomic(const QString& path, const QJsonArray& entries);
    QJsonArray getGlobalLog() const;
    QSet<QString> collectUsedFaceIdsFromAllDrives();

    // === Face Detection / Embedding APIs ===
    std::vector<QRect> detectFaces(const cv::Mat& image);
    std::vector<QRect> detectFaces(const QImage& image);  // NEW
    std::vector<float> getFaceEmbedding(const cv::Mat& image, const QRect& faceRect);
    std::vector<float> getJitteredEmbedding(const cv::Mat& image, const QRect& faceRect);  // NEW
    bool isMatchingFace(const cv::Mat& face1, const cv::Mat& face2, float threshold = 0.6f);  // NEW

signals:
    void faceLogged(const QString& id, const QString& thumbPath);

private:
    void loadModels();
    QJsonArray loadFaceLog(const QString& path) const;

    QString predictorPath;
    QString recognitionModelPath;
    class Impl;
    Impl* impl;
};
