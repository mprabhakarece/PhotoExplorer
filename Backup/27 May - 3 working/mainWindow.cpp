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
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

QPixmap getCachedThumbnail(const QString &imagePath, QSize size) {
    QFileInfo info(imagePath);

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

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
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

    goHome();
}

MainWindow::~MainWindow() {}

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

    std::sort(entries.begin(), entries.end(), [](const QFileInfo& a, const QFileInfo& b) {
        if (a.isDir() != b.isDir()) return a.isDir();
        return a.fileName().toLower() < b.fileName().toLower();
    });

    for (const QFileInfo &entry : entries) {
        QListWidgetItem *item = nullptr;

        if (entry.isDir()) {
            // Check if folder contains at least one image (recursively)
            QDirIterator it(entry.absoluteFilePath(),
                            QStringList() << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp",
                            QDir::Files,
                            QDirIterator::Subdirectories);

            if (!it.hasNext()) continue; // Skip folder with no images

            // ✅ Just use default folder icon without overlay
            item = new QListWidgetItem(iconProvider.icon(entry), entry.fileName());
        } else {
            QString ext = entry.suffix().toLower();
            if (!(ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp"))
                continue;

            // Show placeholder first, then load thumbnail async
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

void MainWindow::refresh() {
    if (currentPath.isEmpty()) {
        goHome();
        return;
    }

    // Reload folder view
    navigateTo(currentPath);

    // === Start face recognition and listing in faceList ===
    faceList->clear();
    knownEmbeddings.clear();
    knownFaceThumbs.clear();

    QDirIterator it(currentPath,
                    QStringList() << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp",
                    QDir::Files,
                    includeSubfolders ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags);

    while (it.hasNext()) {
        QString path = it.next();
        cv::Mat matBGR = cv::imread(path.toStdString());
        if (matBGR.empty()) continue;

        auto faces = faceDetector.detectFaces(matBGR);
        qDebug() << "🧠" << faces.size() << "face(s) found in:" << path;

        std::vector<std::pair<QRect, std::vector<float>>> imageEmbeddings;

        for (const QRect& rect : faces) {
            auto embedding = faceDetector.getJitteredEmbedding(matBGR, rect);
            if (!embedding.empty())
                imageEmbeddings.emplace_back(rect, embedding);
        }

        // Now compare to global knownEmbeddings
        for (const auto& [rect, embedding] : imageEmbeddings) {
            bool matched = false;
            for (const auto& known : knownEmbeddings) {
                if (isSimilarFace(embedding, known, 0.6f)) {
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                qDebug() << "➕ New face added from:" << path << "Rect:" << rect;
                knownEmbeddings.push_back(embedding);

                cv::Rect roi(rect.x(), rect.y(), rect.width(), rect.height());
                roi &= cv::Rect(0, 0, matBGR.cols, matBGR.rows);  // clamp

                cv::Mat faceMat = matBGR(roi).clone();
                cv::resize(faceMat, faceMat, cv::Size(64, 64));
                QImage faceImg(faceMat.data, faceMat.cols, faceMat.rows, faceMat.step, QImage::Format_BGR888);
                QPixmap thumb = QPixmap::fromImage(faceImg.copy());

                QListWidgetItem* item = new QListWidgetItem(QIcon(thumb), "");
                item->setToolTip(path);
                faceList->addItem(item);
            } else {
                qDebug() << "✅ Face matched existing person, skipped:" << path << "Rect:" << rect;
            }
        }

    }

    statusBar()->showMessage(QString("✔️ Detected %1 unique people").arg(knownEmbeddings.size()), 3000);
}



bool MainWindow::isSimilarFace(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
    if (a.size() != b.size()) return false;
    float distSq = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        distSq += diff * diff;
    }
    float dist = std::sqrt(distSq);
    return dist < threshold;  // e.g. threshold = 0.6 like Dlib
}


