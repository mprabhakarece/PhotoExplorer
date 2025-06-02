#ifndef FACEINDEXER_H
#define FACEINDEXER_H

#include <QString>
#include <QList>
#include <QRect>
#include <vector>
#include "FaceTypes.h"

class FaceIndexer {
public:
    FaceIndexer();

    // Store a single detected face in the global DB
    bool saveFaceEntry(const QString& imagePath, const QRect& faceRect,
                       const std::vector<float>& embedding, float quality, qint64 mtime);

    // Fetch all face entries under a given folder path (recursively)
    QList<FaceEntry> getFaceEntriesInFolder(const QString& folderPath);

    // Retrieve face entries by shared global ID
    QList<FaceEntry> getFaceEntriesByGlobalId(const QString& globalId);

    // Avoid reprocessing already-seen image if unchanged
    bool faceAlreadyProcessed(const QString& imagePath, qint64 mtime);

    // Match embedding to existing global ID or assign new one
    QString assignOrFindGlobalId(const std::vector<float>& embedding);

};

#endif // FACEINDEXER_H
