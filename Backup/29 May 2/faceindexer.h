#ifndef FACEINDEXER_H
#define FACEINDEXER_H

#include <QString>
#include <QJsonArray>
#include <QRect>
#include <vector>

class FaceIndexer {
public:
    QJsonArray getFaceLog(const QString& folder);
    bool hasFaceData(const QString& imagePath, const QJsonArray& log);
    void addFaceData(const QString& imagePath,
                     const std::vector<QRect>& rects,
                     const std::vector<std::vector<float>>& embeddings,
                     const std::vector<double>& symmetryScores,
                     const std::vector<double>& focusScores);
};

#endif // FACEINDEXER_H
