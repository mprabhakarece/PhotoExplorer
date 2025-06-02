#include "faceindexer.h"
#include "FaceDatabaseManager.h"
#include <QDebug>
#include <QDirIterator>

FaceIndexer::FaceIndexer() {
    // Nothing needed for now; DB is initialized in mainWindow or via singleton
}

bool FaceIndexer::saveFaceEntry(const QString& imagePath, const QRect& faceRect,
                                const std::vector<float>& embedding, float quality, qint64 mtime)
{
    return FaceDatabaseManager::instance().addFace(imagePath, faceRect, embedding, quality, mtime);
}

QList<FaceEntry> FaceIndexer::getFaceEntriesInFolder(const QString& folderPath)
{
    return FaceDatabaseManager::instance().getFacesForFolder(folderPath);
}

QList<FaceEntry> FaceIndexer::getFaceEntriesByGlobalId(const QString& globalId)
{
    return FaceDatabaseManager::instance().getFacesByGlobalId(globalId);
}

bool FaceIndexer::faceAlreadyProcessed(const QString& imagePath, qint64 mtime)
{
    return FaceDatabaseManager::instance().faceAlreadyProcessed(imagePath, mtime);
}

QString FaceIndexer::assignOrFindGlobalId(const std::vector<float>& embedding)
{
    return FaceDatabaseManager::instance().assignOrFindGlobalID(embedding);
}

