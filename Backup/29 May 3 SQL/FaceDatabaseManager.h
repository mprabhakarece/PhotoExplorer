#ifndef FACEDATABASEMANAGER_H
#define FACEDATABASEMANAGER_H

#include <QSqlDatabase>
#include <QSqlQuery>
#include <QRect>
#include <QString>
#include <QList>
#include <vector>
#include"FaceTypes.h"

class FaceDatabaseManager {
public:
    static FaceDatabaseManager& instance();
    bool open(const QString& dbPath);
    void ensureTables();

    bool addFace(const QString& imagePath, const QRect& rect, const std::vector<float>& embedding, float quality, qint64 mtime);
    QList<FaceEntry> getFacesForFolder(const QString& folderPath);
    QList<FaceEntry> getFacesByGlobalId(const QString& globalId);
    std::vector<float> getEmbeddingById(int id);
    bool faceAlreadyProcessed(const QString& imagePath, qint64 mtime);

    QString assignOrFindGlobalID(const std::vector<float>& embedding);

private:
    QSqlDatabase db;
    FaceDatabaseManager();
    void openDatabase();
    QByteArray embeddingToBlob(const std::vector<float>& emb);
    std::vector<float> blobToEmbedding(const QByteArray& blob);
};

#endif // FACEDATABASEMANAGER_H
