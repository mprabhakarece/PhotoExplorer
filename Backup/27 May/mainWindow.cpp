#include "mainwindow.h"
#include <QToolBar>
#include <QDir>
#include <QFileInfo>
#include <QFileIconProvider>
#include <QDirIterator>
#include <QVBoxLayout>
#include <QDebug>
#include <QPainter>
#include <QtConcurrent/QtConcurrent>
#include <QSplitter>
#include <QPushButton>
#include <QCheckBox>
#include <opencv2/imgcodecs.hpp>
#include <QStatusBar>

QPixmap getCachedThumbnail(const QString &imagePath, QSize size) {
    QFileInfo info(imagePath);

    //Skip any file that is inside a `.cache` folder
    QString lowerPath = info.filePath().toLower();

    QDir cacheDir(info.dir().absolutePath() + "/.cache/thumbnails");
    if (!cacheDir.exists()) cacheDir.mkpath(".");

    QString thumbPath = cacheDir.absoluteFilePath(info.fileName() + ".thumb");

    if (QFile::exists(thumbPath)) {
        QPixmap cached(thumbPath);
        if (!cached.isNull()) return cached;
    }

    QPixmap original(imagePath);
    if (original.isNull()) return QPixmap();

    QPixmap thumb = original.scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    thumb.save(thumbPath, "JPG");
    return thumb;
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {
    setWindowTitle("Photo Explorer");
    resize(1200, 800);

    QToolBar *toolbar = addToolBar("Navigation");

    QWidget *toolbarWidget = new QWidget(this);
    QHBoxLayout *toolbarLayout = new QHBoxLayout(toolbarWidget);
    toolbarLayout->setContentsMargins(5, 2, 5, 2);
    toolbarLayout->setSpacing(8);

    QPushButton *backButton = new QPushButton("Back");
    QPushButton *homeButton = new QPushButton("Home");
    QPushButton *refreshButton = new QPushButton("Refresh");
    QPushButton *clearCacheButton = new QPushButton("Clear Cache");

    QCheckBox *includeSubfoldersCheckbox = new QCheckBox("Include Subfolders");
    pathLabel = new QLabel("This PC");

    backButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    homeButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    clearCacheButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    refreshButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    includeSubfoldersCheckbox->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    pathLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    toolbarLayout->addWidget(backButton);
    toolbarLayout->addWidget(homeButton);
    toolbarLayout->addWidget(pathLabel);
    toolbarLayout->addWidget(includeSubfoldersCheckbox);
    toolbarLayout->addWidget(clearCacheButton);
    toolbarLayout->addWidget(refreshButton);

    toolbar->addWidget(toolbarWidget);
    toolbarWidget->setMinimumHeight(36);

    connect(backButton, &QPushButton::clicked, this, &MainWindow::goBack);
    connect(homeButton, &QPushButton::clicked, this, &MainWindow::goHome);
    connect(includeSubfoldersCheckbox, &QCheckBox::toggled, this, [=](bool checked) {
        includeSubfolders = checked;
        refresh();
    });
    connect(clearCacheButton, &QPushButton::clicked, this, [=]() {
        QString thumbPath = currentPath + "/.cache/faces";
        QDir dir(thumbPath);
        if (dir.exists()) {
            dir.setFilter(QDir::Files);
            for (const QFileInfo &file : dir.entryInfoList()) {
                QFile::remove(file.absoluteFilePath());
            }
            statusBar()->showMessage("Face cache cleared.", 3000);
        }
    });
    connect(refreshButton, &QPushButton::clicked, this, &MainWindow::refresh);

    QSplitter *mainSplitter = new QSplitter(this);
    setCentralWidget(mainSplitter);

    faceList = new QListWidget(this);
    faceList->setViewMode(QListView::IconMode);
    faceList->setIconSize(QSize(64, 64));
    faceList->setResizeMode(QListView::Adjust);
    faceList->setMovement(QListView::Static);
    faceList->setSpacing(10);
    faceList->setMaximumWidth(200);
    mainSplitter->addWidget(faceList);

    stack = new QStackedWidget(this);
    mainSplitter->addWidget(stack);

    driveList = new QListWidget(this);
    driveList->setViewMode(QListView::IconMode);
    driveList->setIconSize(QSize(64, 64));
    driveList->setResizeMode(QListView::Adjust);
    driveList->setMovement(QListView::Static);
    driveList->setSpacing(20);
    driveList->setSelectionMode(QAbstractItemView::SingleSelection);
    driveList->setFocusPolicy(Qt::StrongFocus);
    driveList->setFocus();
    stack->addWidget(driveList);

    connect(driveList, &QListWidget::itemClicked, this, [this](QListWidgetItem *item) {
        QString path = item->data(Qt::UserRole).toString();
        navigateTo(path);
    });

    folderView = new QListWidget(this);
    folderView->setViewMode(QListView::IconMode);
    folderView->setIconSize(QSize(128, 128));
    folderView->setGridSize(QSize(140, 160));
    folderView->setSpacing(10);
    folderView->setResizeMode(QListView::Adjust);
    folderView->setMovement(QListView::Static);
    folderView->setSelectionMode(QAbstractItemView::SingleSelection);
    folderView->setFocusPolicy(Qt::StrongFocus);
    folderView->setFocus();
    stack->addWidget(folderView);

    connect(folderView, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *item) {
        QFileInfo info(item->data(Qt::UserRole).toString());
        if (info.isDir()) {
            navigateTo(info.absoluteFilePath());
        }
    });

    faceIndexer = new FaceIndexer("models");
    goHome();
}

void MainWindow::goHome() {
    pathLabel->setText("This PC");
    currentPath.clear();
    navHistory.clear();
    loadDrives();
    stack->setCurrentWidget(driveList);
}

void MainWindow::goBack() {
    if (navHistory.size() >= 2) {
        navHistory.removeLast();
        navigateTo(navHistory.last());
    } else {
        goHome();
    }
}

void MainWindow::navigateTo(const QString &path) {
    if (!QDir(path).exists()) return;
    currentPath = path;
    pathLabel->setText(path);
    navHistory.append(path);
    loadFolder(path);
    stack->setCurrentWidget(folderView);
}

void MainWindow::loadDrives() {
    driveList->clear();
    QFileInfoList drives = QDir::drives();
    QFileIconProvider iconProvider;
    for (const QFileInfo &drive : drives) {
        QListWidgetItem *item = new QListWidgetItem(iconProvider.icon(drive), drive.absoluteFilePath());
        item->setData(Qt::UserRole, drive.absoluteFilePath());
        driveList->addItem(item);
    }
}

void MainWindow::loadFolder(const QString &path) {
    folderView->clear();
    QDir dir(path);
    QFileInfoList entries = dir.entryInfoList(QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot);
    QFileIconProvider iconProvider;

    faceList->clear();
    QSet<QString> seenIds;

    if (faceIndexer) {
        QJsonArray faceLog = faceIndexer->getFaceLogRaw(currentPath);
        for (const QJsonValue& val : faceLog) {
            QJsonObject obj = val.toObject();
            QString id = obj["id"].toString();
            if (seenIds.contains(id)) continue;
            seenIds.insert(id);

            QString thumbPath = obj.contains("thumb") ? obj["thumb"].toString() : obj["image"].toString();
            QPixmap facePix;
            if (facePix.load(thumbPath)) {
                facePix = facePix.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation);
                QListWidgetItem* item = new QListWidgetItem(QIcon(facePix), id);
                item->setData(Qt::UserRole, id);
                faceList->addItem(item);
            }
        }
    }

    std::sort(entries.begin(), entries.end(), [](const QFileInfo& a, const QFileInfo& b) {
        if (a.isDir() != b.isDir()) return a.isDir();
        return a.fileName().toLower() < b.fileName().toLower();
    });

    for (const QFileInfo &entry : entries) {
        if (entry.isDir()) {
            bool hasImages = false;
            QDirIterator it(entry.absoluteFilePath(), QStringList() << "*.jpg" << "*.png" << "*.bmp", QDir::Files, QDirIterator::Subdirectories);
            if (it.hasNext()) hasImages = true;
            if (!hasImages) continue;
        } else if (!entry.fileName().endsWith(".jpg", Qt::CaseInsensitive) &&
                   !entry.fileName().endsWith(".png", Qt::CaseInsensitive) &&
                   !entry.fileName().endsWith(".bmp", Qt::CaseInsensitive)) {
            continue;
        }

        QListWidgetItem *item = nullptr;
        if (entry.isDir()) {
            item = new QListWidgetItem(iconProvider.icon(entry), entry.fileName());
        } else {
            item = new QListWidgetItem(QIcon(":/icons/placeholder.png"), entry.fileName());
            QFuture<void> _ = QtConcurrent::run([=]() {
                QPixmap thumb = getCachedThumbnail(entry.absoluteFilePath(), QSize(128, 128));
                if (!thumb.isNull()) {
                    QMetaObject::invokeMethod(folderView, [=]() {
                        item->setIcon(QIcon(thumb));
                    }, Qt::QueuedConnection);
                }
            });
        }
        item->setToolTip(entry.absoluteFilePath());
        item->setData(Qt::UserRole, entry.absoluteFilePath());
        folderView->addItem(item);
    }
}

QIcon MainWindow::generateFolderPreviewIcon(const QString &folderPath, QSize thumbSize) {
    QStringList imageFiles = QDir(folderPath).entryList(QStringList() << "*.jpg" << "*.png" << "*.bmp", QDir::Files, QDir::Name);
    if (imageFiles.isEmpty()) return QIcon(":/icons/folder.png");

    QString firstImagePath = folderPath + "/" + imageFiles.first();
    QPixmap base(":/icons/folder.png");
    QPixmap overlay(firstImagePath);
    if (base.isNull() || overlay.isNull()) return QIcon(":/icons/folder.png");

    overlay = overlay.scaled(thumbSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    QPixmap result = base;
    QPainter painter(&result);
    QRect overlayRect(base.width() * 0.25, base.height() * 0.25, base.width() * 0.5, base.height() * 0.5);
    painter.drawPixmap(overlayRect, overlay);
    painter.end();
    return QIcon(result);
}

void MainWindow::refresh() {
    if (currentPath.isEmpty()) {
        goHome();
        return;
    }

    // === 1. Update facelog for current or subfolders ===
    if (faceIndexer) {
        if (includeSubfolders) {
            QDirIterator it(currentPath, QDir::Dirs | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
            while (it.hasNext()) {
                QString subfolder = it.next();
                faceIndexer->buildOrUpdateFaceLog(subfolder);
            }
        } else {
            faceIndexer->buildOrUpdateFaceLog(currentPath);
        }
    }

    // === 2. Reload the UI for this folder ===
    navigateTo(currentPath);

    // === 3. Run global cleanup of unused face IDs in background ===
    QFuture<void> cleanupFuture = QtConcurrent::run([this]() {
        QSet<QString> usedIds = faceIndexer->collectUsedFaceIdsFromAllDrives();

        QString globalLogPath = "C:/PhotoExplorer/faces/global_facelog.json";
        QJsonArray globalLog = faceIndexer->getFaceLogRaw("C:/PhotoExplorer/faces");
        QJsonArray cleaned;

        for (const auto& val : globalLog) {
            QJsonObject obj = val.toObject();
            QString id = obj["id"].toString();
            if (usedIds.contains(id)) {
                cleaned.append(obj);
            } else {
                qDebug() << "🗑️ Removing unused global face ID:" << id;
            }
        }

        faceIndexer->saveFaceLog(globalLogPath, cleaned);
        qDebug() << "✅ Global face ID cleanup completed. Remaining IDs:" << cleaned.size();
    });
}

