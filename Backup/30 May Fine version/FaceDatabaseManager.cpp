#include "FaceDatabaseManager.h"
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QVariant>
#include <QSqlRecord>
#include <QDir>
#include <QCoreApplication>
#include <QThread>

FaceDatabaseManager::FaceDatabaseManager() {
    QString appPath = QCoreApplication::applicationDirPath();
    QString dbPath = appPath + "/.cache/face_database.sqlite";

    QDir cacheDir(QDir::currentPath() + "/.cache");
    if (!cacheDir.exists()) {
        cacheDir.mkpath(".");
    }

    db = QSqlDatabase::addDatabase("QSQLITE");
    db.setDatabaseName(dbPath);

    if (!db.open()) {
        qCritical() << "❌ Failed to open SQLite database:" << db.lastError().text();
        return;
    }

    ensureTables();  // ✅ USE CORRECT TABLE SETUP
}


bool FaceDatabaseManager::open(const QString& dbPath) {
    db = QSqlDatabase::addDatabase("QSQLITE");
    db.setDatabaseName(dbPath);
    if (!db.open()) {
        qWarning() << "❌ Failed to open DB:" << db.lastError().text();
        return false;
    }
    ensureTables();
    return true;
}

void FaceDatabaseManager::ensureTables() {
    QSqlQuery q(getThreadDb());

    q.exec(R"(
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            face_rect TEXT,
            embedding BLOB,
            global_id TEXT,
            quality REAL,
            mtime INTEGER
        )
    )");

    q.exec(R"(
        CREATE TABLE IF NOT EXISTS global_faces (
            global_id TEXT PRIMARY KEY,
            label TEXT,
            avg_embedding BLOB,
            count INTEGER
        )
    )");

    q.exec(R"(
        CREATE TABLE IF NOT EXISTS scan_log (
            folder_path TEXT PRIMARY KEY,
            mtime INTEGER,
            face_count INTEGER
        )
    )");
}

bool FaceDatabaseManager::addFace(const QString& imagePath, const QRect& rect,
                                  const std::vector<float>& embedding, float quality, qint64 mtime)
{
    QSqlQuery q(FaceDatabaseManager::getThreadDb());
    q.prepare(R"(
        INSERT INTO face_embeddings (image_path, face_rect, embedding, global_id, quality, mtime)
        VALUES (?, ?, ?, ?, ?, ?)
    )");
    q.addBindValue(imagePath);
    q.addBindValue(QString("[%1,%2,%3,%4]").arg(rect.x()).arg(rect.y()).arg(rect.width()).arg(rect.height()));
    q.addBindValue(embeddingToBlob(embedding));
    q.addBindValue("");  // empty global_id for now
    q.addBindValue(quality);
    q.addBindValue(mtime);

    if (!q.exec()) {
        qWarning() << "❌ Failed to insert face:" << q.lastError().text();
        return false;
    }
    return true;
}

bool FaceDatabaseManager::faceAlreadyProcessed(const QString& imagePath, qint64 mtime)
{
    QSqlQuery q(FaceDatabaseManager::getThreadDb());
    q.prepare("SELECT COUNT(*) FROM face_embeddings WHERE image_path = ? AND mtime = ?");
    q.addBindValue(imagePath);
    q.addBindValue(mtime);
    if (q.exec() && q.next()) {
        return q.value(0).toInt() > 0;
    }
    return false;
}

QList<FaceEntry> FaceDatabaseManager::getFacesForFolder(const QString& folderPath)
{
    QList<FaceEntry> list;
    QSqlQuery q(FaceDatabaseManager::getThreadDb());
    QString pathPrefix = folderPath.endsWith("/") ? folderPath : folderPath + "/";
    q.prepare("SELECT id, image_path, face_rect, global_id, quality FROM face_embeddings WHERE image_path LIKE ?");
    q.addBindValue(pathPrefix + "%");

    if (q.exec()) {
        while (q.next()) {
            FaceEntry entry;
            entry.id = q.value(0).toInt();
            entry.imagePath = q.value(1).toString();
            QStringList rectParts = q.value(2).toString().remove("[").remove("]").split(",");
            if (rectParts.size() == 4) {
                entry.faceRect = QRect(rectParts[0].toInt(), rectParts[1].toInt(), rectParts[2].toInt(), rectParts[3].toInt());
            }
            entry.globalId = q.value(3).toString();
            entry.quality = q.value(4).toFloat();
            list.append(entry);
        }
    }
    return list;
}

QList<FaceEntry> FaceDatabaseManager::getFacesByGlobalId(const QString& globalId)
{
    QList<FaceEntry> list;
    QSqlQuery q(FaceDatabaseManager::getThreadDb());
    q.prepare("SELECT id, image_path, face_rect, quality FROM face_embeddings WHERE global_id = ?");
    q.addBindValue(globalId);

    if (q.exec()) {
        while (q.next()) {
            FaceEntry entry;
            entry.id = q.value(0).toInt();
            entry.imagePath = q.value(1).toString();
            QStringList rectParts = q.value(2).toString().remove("[").remove("]").split(",");
            if (rectParts.size() == 4) {
                entry.faceRect = QRect(rectParts[0].toInt(), rectParts[1].toInt(), rectParts[2].toInt(), rectParts[3].toInt());
            }
            entry.globalId = globalId;
            entry.quality = q.value(3).toFloat();
            list.append(entry);
        }
    }
    return list;
}

QByteArray FaceDatabaseManager::embeddingToBlob(const std::vector<float>& emb)
{
    QByteArray blob;
    blob.resize(static_cast<int>(emb.size() * sizeof(float)));
    memcpy(blob.data(), emb.data(), blob.size());
    return blob;
}

std::vector<float> FaceDatabaseManager::blobToEmbedding(const QByteArray& blob)
{
    std::vector<float> result(blob.size() / sizeof(float));
    memcpy(result.data(), blob.constData(), blob.size());
    return result;
}

std::vector<float> FaceDatabaseManager::getEmbeddingById(int id) {
    std::vector<float> embedding;

    QSqlQuery query(getThreadDb());

    query.prepare("SELECT embedding FROM face_embeddings WHERE id = ?");
    query.addBindValue(id);

    if (query.exec() && query.next()) {
        QByteArray blob = query.value(0).toByteArray();
        embedding = blobToEmbedding(blob);
    } else {
        qWarning() << "⚠️ Failed to fetch embedding for ID:" << id << query.lastError().text();
    }

    return embedding;
}

QString FaceDatabaseManager::assignOrFindGlobalID(const std::vector<float>& embedding) {
    QByteArray embBlob = embeddingToBlob(embedding);
    QString matchedId;

    QSqlQuery query(getThreadDb());
    query.prepare("SELECT global_id, avg_embedding FROM global_faces");
    if (query.exec()) {
        while (query.next()) {
            int existingId = query.value(0).toInt();
            QByteArray existingBlob = query.value(1).toByteArray();
            std::vector<float> existingEmb = blobToEmbedding(existingBlob);

            // Compare with cosine similarity or L2 distance
            float distSq = 0.0f;
            for (size_t i = 0; i < embedding.size(); ++i) {
                float diff = embedding[i] - existingEmb[i];
                distSq += diff * diff;
            }
            float dist = std::sqrt(distSq);
            if (dist < 0.5f) {
                matchedId = QString::number(existingId);
                break;
            }
        }
    }

    if (matchedId.isEmpty()) {
        // Generate new ID (or insert row to get ID)
        QSqlQuery insert(getThreadDb());
        insert.prepare("INSERT INTO global_faces (avg_embedding) VALUES (?)");
        insert.addBindValue(embBlob);
        if (insert.exec()) {
            matchedId = QString::number(insert.lastInsertId().toInt());
        } else {
            qWarning() << "❌ Failed to insert global face ID:" << insert.lastError().text();
        }
    }

    return matchedId;
}

FaceDatabaseManager& FaceDatabaseManager::instance() {
    static FaceDatabaseManager inst;
    return inst;
}

QSqlDatabase FaceDatabaseManager::getThreadDb() {
    QString connName = QString("face_db_%1").arg(reinterpret_cast<quintptr>(QThread::currentThread()));
    if (!QSqlDatabase::contains(connName)) {
        QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", connName);
        db.setDatabaseName(QCoreApplication::applicationDirPath() + "/.cache/face_database.sqlite");
        if (!db.open()) {
            qWarning() << "❌ Failed to open SQLite DB in thread:" << db.lastError().text();
        } else {
            qDebug() << "✅ Opened thread-safe DB for:" << connName;
        }
    }
    return QSqlDatabase::database(connName);
}

QList<FaceEntry> FaceDatabaseManager::getFaceEntriesInFolder(const QString& folderPath) {
    QList<FaceEntry> result;
    QSqlQuery query(db);

    QString modPath = folderPath;
    modPath.replace("\\", "/");

    // Only entries that are directly inside the folder (not in subfolders)
    query.prepare(R"(
        SELECT id, image_path, face_rect, global_id, quality
        FROM face_embeddings
        WHERE image_path LIKE :folder || '/%' AND image_path NOT LIKE :folder || '/%/%'
    )");

    query.bindValue(":folder", modPath);

    if (query.exec()) {
        while (query.next()) {
            FaceEntry entry;
            entry.id = query.value(0).toInt();
            entry.imagePath = query.value(1).toString();

            QStringList parts = query.value(2).toString().remove("[").remove("]").split(",");
            if (parts.size() == 4) {
                entry.faceRect = QRect(parts[0].toInt(), parts[1].toInt(), parts[2].toInt(), parts[3].toInt());
            }

            entry.globalId = query.value(3).toString();
            entry.quality = query.value(4).toFloat();

            result.append(entry);
        }
    } else {
        qWarning() << "❌ Query failed in getFaceEntriesInFolder:" << query.lastError().text();
    }

    return result;
}

QList<FaceEntry> FaceDatabaseManager::getFaceEntriesInSubtree(const QString& rootPath) {
    QList<FaceEntry> result;
    QSqlQuery query(db);

    QString modPath = rootPath;
    modPath.replace("\\", "/");

    query.prepare(R"(
        SELECT id, image_path, face_rect, global_id, quality
        FROM face_embeddings
        WHERE image_path LIKE :root || '/%'
    )");

    query.bindValue(":root", modPath);

    if (query.exec()) {
        while (query.next()) {
            FaceEntry entry;
            entry.id = query.value(0).toInt();
            entry.imagePath = query.value(1).toString();

            QStringList parts = query.value(2).toString().remove("[").remove("]").split(",");
            if (parts.size() == 4) {
                entry.faceRect = QRect(parts[0].toInt(), parts[1].toInt(), parts[2].toInt(), parts[3].toInt());
            }

            entry.globalId = query.value(3).toString();
            entry.quality = query.value(4).toFloat();

            result.append(entry);
        }
    } else {
        qWarning() << "❌ Query failed in getFaceEntriesInSubtree:" << query.lastError().text();
    }

    return result;
}
