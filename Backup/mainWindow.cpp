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

QPixmap getCachedThumbnail(const QString &imagePath, QSize size) {
    QFileInfo info(imagePath);
    QDir cacheDir(info.dir().absolutePath() + "/.cache/thumbnails");
    if (!cacheDir.exists()) cacheDir.mkpath(".");

    QString thumbPath = cacheDir.absoluteFilePath(info.fileName() + ".thumb");

    // Load from cache if available
    if (QFile::exists(thumbPath)) {
        QPixmap cached(thumbPath);
        if (!cached.isNull()) return cached;
    }

    // Generate and save thumbnail
    QPixmap original(imagePath);
    if (original.isNull()) return QPixmap();

    QPixmap thumb = original.scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    thumb.save(thumbPath, "JPG");
    return thumb;
}
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("Photo Explorer");
    resize(1200, 800);

    // Toolbar
    QToolBar *toolbar = addToolBar("Navigation");
    QAction *backAction = toolbar->addAction("Back");
    QAction *homeAction = toolbar->addAction("Home");
    QAction *refreshAction = toolbar->addAction("Refresh");

    pathLabel = new QLabel("This PC");
    toolbar->addWidget(pathLabel);

    connect(backAction, &QAction::triggered, this, &MainWindow::goBack);
    connect(homeAction, &QAction::triggered, this, &MainWindow::goHome);
    connect(refreshAction, &QAction::triggered, this, &MainWindow::refresh);

    // Stacked views
    stack = new QStackedWidget(this);
    setCentralWidget(stack);

    // Drive list view
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

    // Folder view
    folderView = new QListWidget(this);
    folderView->setViewMode(QListView::IconMode);
    folderView->setIconSize(QSize(128, 128));
    folderView->setGridSize(QSize(140, 160));
    folderView->setSpacing(10);
    folderView->setResizeMode(QListView::Adjust);
    folderView->setMovement(QListView::Static);
    folderView->setSpacing(10);
    folderView->setSelectionMode(QAbstractItemView::SingleSelection);
    folderView->setFocusPolicy(Qt::StrongFocus);
    folderView->setFocus();
    folderView->setSortingEnabled(true);
    stack->addWidget(folderView);

    connect(folderView, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *item) {
        QFileInfo info(item->data(Qt::UserRole).toString());
        if (info.isDir()) {
            navigateTo(info.absoluteFilePath());
        }
    });

    // Show home view
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

void MainWindow::refresh() {
    if (!currentPath.isEmpty()) {
        navigateTo(currentPath);
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

    std::sort(entries.begin(), entries.end(), [](const QFileInfo& a, const QFileInfo& b) {
        if (a.isDir() != b.isDir()) return a.isDir();  // Folders first
        return a.fileName().toLower() < b.fileName().toLower();
    });

    for (const QFileInfo &entry : entries) {
        if (entry.isDir()) {
            // Check if the folder contains any image
            bool hasImages = false;
            QDirIterator it(entry.absoluteFilePath(),
                            QStringList() << "*.jpg" << "*.png" << "*.bmp",
                            QDir::Files,
                            QDirIterator::Subdirectories);
            if (it.hasNext())
                hasImages = true;

            if (!hasImages) continue;
              // Skip folders without images
        }
        else if (!entry.fileName().endsWith(".jpg", Qt::CaseInsensitive) &&
                 !entry.fileName().endsWith(".png", Qt::CaseInsensitive) &&
                 !entry.fileName().endsWith(".bmp", Qt::CaseInsensitive)) {
            continue;  // Skip non-image files
        }


        QListWidgetItem *item = nullptr;

        if (entry.isDir()) {
            // folder → use default icon
            item = new QListWidgetItem(iconProvider.icon(entry), entry.fileName());
        } else {
            // image → async thumbnail
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

QIcon generateFolderPreviewIcon(const QString &folderPath, QSize thumbSize) {
    QStringList imageFiles = QDir(folderPath).entryList(
        QStringList() << "*.jpg" << "*.png" << "*.bmp",
        QDir::Files, QDir::Name);

    if (imageFiles.isEmpty())
        return QIcon(":/icons/folder.png");  // fallback to plain folder icon

    QString firstImagePath = folderPath + "/" + imageFiles.first();
    QPixmap base(":/icons/folder.png");     // your base folder icon
    QPixmap overlay(firstImagePath);

    if (base.isNull() || overlay.isNull())
        return QIcon(":/icons/folder.png");

    // Resize overlay
    overlay = overlay.scaled(thumbSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Draw overlay on top of folder icon
    QPixmap result = base;
    QPainter painter(&result);

    // ✅ This is the part you were asking about:
    QRect overlayRect(base.width() * 0.25, base.height() * 0.25,
                      base.width() * 0.5, base.height() * 0.5);

    painter.drawPixmap(overlayRect, overlay);  // 🟡 This replaces: drawPixmap(8, 8, overlay)
    painter.end();

    return QIcon(result);
}
