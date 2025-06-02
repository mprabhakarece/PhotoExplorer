#include "faceindexer.h"

#include <QFile>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>

QJsonArray FaceIndexer::getFaceLog(const QString& folder) {
    QFile file(folder + "/.cache/facelog.json");
    if (!file.exists()) return QJsonArray();
    if (!file.open(QIODevice::ReadOnly)) return QJsonArray();

    QByteArray data = file.readAll();
    QJsonDocument doc = QJsonDocument::fromJson(data);
    return doc.isArray() ? doc.array() : QJsonArray();
}

bool FaceIndexer::hasFaceData(const QString& imagePath, const QJsonArray& log) {
    QString name = QFileInfo(imagePath).fileName();
    for (const QJsonValue& val : log) {
        QJsonObject obj = val.toObject();
        if (obj["image"].toString() == name)
            return true;
    }
    return false;
}

void FaceIndexer::addFaceData(const QString& imagePath,
                              const std::vector<QRect>& rects,
                              const std::vector<std::vector<float>>& embeddings,
                              const std::vector<double>& symmetryScores,
                              const std::vector<double>& focusScores) {
    QString folder = QFileInfo(imagePath).absolutePath();
    QDir cacheDir(folder + "/.cache");
    if (!cacheDir.exists()) cacheDir.mkpath(".");

    QJsonArray log = getFaceLog(folder);

    QJsonObject entry;
    entry["image"] = QFileInfo(imagePath).fileName();

    QJsonArray faceList;
    for (int i = 0; i < rects.size(); ++i) {
        const QRect& r = rects[i];
        const auto& emb = embeddings[i];
        double sym = symmetryScores[i];
        double foc = focusScores[i];

        QJsonObject face;
        face["rect"] = QJsonArray{r.x(), r.y(), r.width(), r.height()};

        QJsonArray embed;
        for (float val : emb)
            embed.append(val);
        face["embedding"] = embed;

        face["symmetry"] = sym;
        face["focus"] = foc;

        faceList.append(face);
    }

    entry["faces"] = faceList;
    log.append(entry);

    QFile out(folder + "/.cache/facelog.json");
    if (out.open(QIODevice::WriteOnly)) {
        QJsonDocument doc(log);
        out.write(doc.toJson(QJsonDocument::Compact));
        out.close();
    }
}
