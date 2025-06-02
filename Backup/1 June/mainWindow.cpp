#include <QToolBar>
#include <QDir>
#include <QFileInfo>
#include <QFileIconProvider>
#include <QDirIterator>
#include <QVBoxLayout>
#include <QDebug>
#include <QPainter>
#include <QtConcurrent/QtConcurrentRun>
#include <QTimer>
#include <QSplitter>
#include <QPushButton>
#include <QCheckBox>
#include <QStatusBar>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QImageReader>
#include <QInputDialog>
#include <QMenu>
#include <QLineEdit>
#include <QDir>
#include <QMessageBox>
#include <QDesktopServices>
#include <QCoreApplication>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mainwindow.h"
#include "faceindexer.h"
#include "FaceListItemDelegate.h"
#include "FaceDatabaseManager.h"
#include "embeddingUtils.h"

FaceIndexer faceIndexer;


constexpr double matchDIST = 0.5f;

constexpr double goodFocusThreshold = 100.0; // e.g. ideal Laplacian variance
constexpr double focusTolerance = 25.0;      // how far below is still acceptable

enum class ResizeMode {
    Original,
    Fit1024x1024,
    Fit1280x720
};

cv::Mat resizeImageForDetection(const cv::Mat& input, ResizeMode mode,
                                cv::Size& newSize, double& scaleX, double& scaleY) {
    if (mode == ResizeMode::Original) {
        newSize = input.size();
        scaleX = 1.0;
        scaleY = 1.0;
        return input.clone();
    }

    int maxWidth, maxHeight;
    if (mode == ResizeMode::Fit1024x1024) {
        maxWidth = 1024;
        maxHeight = 1024;
    } else { // Fit1280x720
        maxWidth = 1280;
        maxHeight = 720;
    }

    int originalWidth = input.cols;
    int originalHeight = input.rows;

    double scale = std::min((double)maxWidth / originalWidth, (double)maxHeight / originalHeight);

    int resizedWidth = static_cast<int>(originalWidth * scale);
    int resizedHeight = static_cast<int>(originalHeight * scale);

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_AREA);

    scaleX = (double)originalWidth / resizedWidth;
    scaleY = (double)originalHeight / resizedHeight;
    newSize = cv::Size(resizedWidth, resizedHeight);

    return resized;
}

QString virtualCachePath(const QString& actualPath) {
    QString base = QCoreApplication::applicationDirPath() + "/.cache";
    QString relative = actualPath;

#ifdef Q_OS_WIN
    if (relative.length() >= 2 && relative[1] == ':') {
        QString driveLetter = relative.left(1);
        relative.remove(0, 2);  // remove "D:"
        relative = driveLetter + "_/" + relative;
    }
#endif

    relative = QDir::cleanPath(relative);  // ✅ normalize "//" or "../"
    return base + "/" + relative;
}

std::vector<float> normalizeEmbedding(const std::vector<float>& emb) {
    float norm = std::sqrt(std::inner_product(emb.begin(), emb.end(), emb.begin(), 0.0f));
    std::vector<float> result(emb.size());
    if (norm > 0.0f) {
        std::transform(emb.begin(), emb.end(), result.begin(), [=](float v) { return v / norm; });
    }
    return result;
}

cv::Mat resizeToFixedHeight(const cv::Mat& input, cv::Size& newSize, double& scaleX, double& scaleY) {
    const int targetHeight = 768 * 1.5;
    double ratio = static_cast<double>(targetHeight) / input.rows;
    int targetWidth = static_cast<int>(input.cols * ratio);

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_AREA);

    scaleX = static_cast<double>(input.cols) / targetWidth;
    scaleY = static_cast<double>(input.rows) / targetHeight;
    newSize = cv::Size(targetWidth, targetHeight);
    return resized;
}

cv::Mat resizeToFit1280x720(const cv::Mat& input, cv::Size& newSize, double& scaleX, double& scaleY) {
    const int maxWidth = 1280;
    const int maxHeight = 720;

    int originalWidth = input.cols;
    int originalHeight = input.rows;

    double scale = std::min((double)maxWidth / originalWidth, (double)maxHeight / originalHeight);

    int resizedWidth = static_cast<int>(originalWidth * scale);
    int resizedHeight = static_cast<int>(originalHeight * scale);

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(resizedWidth, resizedHeight), 0, 0, cv::INTER_AREA);

    scaleX = (double)originalWidth / resizedWidth;
    scaleY = (double)originalHeight / resizedHeight;
    newSize = cv::Size(resizedWidth, resizedHeight);

    return resized;
}

bool eyesAreOpen(const dlib::full_object_detection& shape) {
    auto eyeOpenness = [&](int top1, int top2, int bottom1, int bottom2) {
        return (shape.part(bottom1).y() + shape.part(bottom2).y()) -
               (shape.part(top1).y() + shape.part(top2).y());
    };

    double leftEye = eyeOpenness(37, 38, 41, 40);  // left eye
    double rightEye = eyeOpenness(43, 44, 47, 46); // right eye

    double eyeOpenScore = (leftEye + rightEye) / 2.0;
    return eyeOpenScore > 4.0;  // adjust threshold if needed
}

double getSymmetryScore(const dlib::full_object_detection& shape) {
    double eyeCenter = (shape.part(36).x() + shape.part(45).x()) / 2.0;
    double noseX = shape.part(30).x();
    return std::abs(eyeCenter - noseX);  // smaller = more frontal
}

double getFocusScore(const cv::Mat& faceMat) {
    cv::Mat gray, lap;
    cv::cvtColor(faceMat, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];  // variance = sharpness
}

struct FaceStats {
    std::vector<float> embedding;
    double symmetry;
    double focus;
    QPixmap thumb;
    QString imagePath;
    int count = 1;  // how many images this person matched
};


std::vector<FaceStats> personList;


QPixmap getCachedThumbnail(const QString &imagePath, QSize size) {
    QFileInfo info(imagePath);
    if (!info.exists()) {
        qWarning() << "❌ Input file does not exist:" << imagePath;
        return QPixmap();
    }

    QString cachePath = virtualCachePath(info.absolutePath()) + "/thumbnails";
    QString thumbPath = QDir(cachePath).absoluteFilePath(info.fileName() + ".thumb");

    qDebug() << "🔍 Thumbnail request for:" << imagePath;
    qDebug() << "📁 Cache folder:" << cachePath;
    qDebug() << "📄 Thumbnail file:" << thumbPath;

    // ✅ Ensure cache folder is created in thread-safe way
    static QMutex cacheMutex;
    {
        QMutexLocker locker(&cacheMutex);
        QDir dir(cachePath);
        if (!dir.exists()) {
            bool created = dir.mkpath(".");
            qDebug() << (created ? "✅ Cache folder created:" : "❌ Failed to create cache folder:") << cachePath;
        }
    }

    // ✅ Load existing thumbnail if available
    if (QFile::exists(thumbPath)) {
        QPixmap cached(thumbPath);
        if (!cached.isNull()) {
            qDebug() << "📦 Loaded cached thumbnail from:" << thumbPath;
            return cached;
        } else {
            qWarning() << "⚠️ Cached .thumb exists but failed to load:" << thumbPath;
        }
    }

    // ✅ Load original image with EXIF rotation
    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    QImage image = reader.read();
    if (image.isNull()) {
        qWarning() << "❌ Failed to load image:" << imagePath << "| Error:" << reader.errorString();
        return QPixmap();
    }

    QPixmap thumb = QPixmap::fromImage(image.scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    bool saved = thumb.save(thumbPath, "JPG");

    if (!saved) {
        qWarning() << "❌ Failed to save thumbnail to:" << thumbPath;
    } else {
        qDebug() << "💾 Saved new thumbnail to:" << thumbPath;
    }

    return thumb;
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("Photo Explorer");
    resize(1200, 800);

    // ==== Top Toolbar ====
    QToolBar *toolbar = addToolBar("Navigation");
    QWidget *toolbarWidget = new QWidget(this);
    QHBoxLayout *toolbarLayout = new QHBoxLayout(toolbarWidget);
    toolbarLayout->setContentsMargins(5, 2, 5, 2);
    toolbarLayout->setSpacing(8);

    // Toolbar buttons
    QPushButton *backButton = new QPushButton("Back");
    QPushButton *homeButton = new QPushButton("Home");
    QPushButton *photoScanButton = new QPushButton("Photo Scan");
    QPushButton *copyButton = new QPushButton("Copy Selected");
    QPushButton *pasteButton = new QPushButton("Paste Here");
    QPushButton *createFolderButton = new QPushButton("New Folder");

    QCheckBox *showHiddenFoldersCheckbox = new QCheckBox("Show Hidden Folders");
    QCheckBox *includeSubfoldersCheckbox = new QCheckBox("Include Subfolders");
    pathLabel = new QLabel("This PC");

    // Add to layout
    toolbarLayout->addWidget(backButton);
    toolbarLayout->addWidget(homeButton);
    toolbarLayout->addWidget(pathLabel);
    toolbarLayout->addWidget(showHiddenFoldersCheckbox);
    toolbarLayout->addStretch();
    toolbarLayout->addWidget(createFolderButton);
    toolbarLayout->addWidget(copyButton);
    toolbarLayout->addWidget(pasteButton);
    toolbarLayout->addWidget(includeSubfoldersCheckbox);
    toolbarLayout->addWidget(photoScanButton);
    toolbar->addWidget(toolbarWidget);
    toolbarWidget->setMinimumHeight(36);

    // ==== Split layout: Face List | Folder Area ====
    QSplitter *mainSplitter = new QSplitter(this);
    setCentralWidget(mainSplitter);

    // === Left Face List ===
    faceList = new QListWidget(this);
    faceList->setViewMode(QListView::IconMode);
    faceList->setIconSize(QSize(84, 84));
    faceList->setGridSize(QSize(90, 110));
    faceList->setSpacing(10);
    faceList->setResizeMode(QListView::Adjust);
    faceList->setMovement(QListView::Static);
    faceList->setMaximumWidth(200);
    faceList->setItemDelegate(new FaceListItemDelegate(FaceListItemDelegate::FaceListMode, this));
    mainSplitter->addWidget(faceList);

    // === Stack: Drives / FolderView ===
    stack = new QStackedWidget(this);
    mainSplitter->addWidget(stack);

    // === Drive List View ===
    driveList = new QListWidget(this);
    driveList->setViewMode(QListView::IconMode);
    driveList->setIconSize(QSize(64, 64));
    driveList->setResizeMode(QListView::Adjust);
    driveList->setMovement(QListView::Static);
    driveList->setSpacing(20);
    driveList->setSelectionMode(QAbstractItemView::SingleSelection);
    driveList->setFocus();
    stack->addWidget(driveList);

    // === Folder View (Right side thumbnails) ===
    folderView = new QListWidget(this);
    folderView->setViewMode(QListView::IconMode);
    folderView->setIconSize(QSize(128, 128));
    folderView->setGridSize(QSize(140, 172));
    folderView->setSpacing(0);
    folderView->setResizeMode(QListView::Adjust);
    folderView->setMovement(QListView::Static);
    folderView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    folderView->setFocus();
    folderView->setItemDelegate(new FaceListItemDelegate(FaceListItemDelegate::FolderViewMode, this));
    folderView->setContextMenuPolicy(Qt::CustomContextMenu);
    stack->addWidget(folderView);

    // ==== Connect UI Logic ====

    // Toolbar buttons
    connect(backButton, &QPushButton::clicked, this, &MainWindow::goBack);
    connect(homeButton, &QPushButton::clicked, this, &MainWindow::goHome);
    connect(photoScanButton, &QPushButton::clicked, this, &MainWindow::refresh);
    connect(copyButton, &QPushButton::clicked, this, &MainWindow::performCopy);
    connect(pasteButton, &QPushButton::clicked, this, &MainWindow::performPaste);

    connect(createFolderButton, &QPushButton::clicked, this, &MainWindow::createNewFolder);

    connect(includeSubfoldersCheckbox, &QCheckBox::toggled, this, [this](bool checked) {
        includeSubfolders = checked;
        statusBar()->showMessage("Include Subfolders " + QString(checked ? "enabled" : "disabled") + ". Click Refresh to apply.", 3000);
        loadFaceListFromDatabase();
    });

    connect(showHiddenFoldersCheckbox, &QCheckBox::toggled, this, [this](bool checked) {
        showHiddenFolders = checked;
        loadFolder(currentPath);
    });

    // Drive list click → navigate
    connect(driveList, &QListWidget::itemClicked, this, [this](QListWidgetItem *item) {
        navigateTo(item->data(Qt::UserRole).toString());
    });

    // Folder view double click → open folder or image
    connect(folderView, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
        QString path = item->data(Qt::UserRole).toString();
        QFileInfo info(path);
        if (info.isDir()) {
            navigateTo(info.absoluteFilePath());
        } else if (info.isFile()) {
            showImagePopup(path);
        }
    });

    connect(folderView, &QListWidget::customContextMenuRequested,
            this, &MainWindow::showFolderViewContextMenu);

    // Face list → update checkboxes in folder view
    connect(faceList, &QListWidget::itemChanged,
            this, &MainWindow::updateFolderViewCheckboxesFromFaceSelection);

    // ==== Startup View ====
    goHome();
}


MainWindow::~MainWindow() {
    scanAbortFlag = true;
    thumbAbortFlag = true;
}


void MainWindow::goHome() {
    abortCurrentScansTemporarily();
    scanAbortFlag = true;
    pathLabel->setText("This PC");
    currentPath.clear();
    navHistory.clear();
    loadDrives();
    stack->setCurrentWidget(driveList);
}

void MainWindow::goBack() {
    abortCurrentScansTemporarily();
    scanAbortFlag = true;
    if (navHistory.size() >= 2) {
        navHistory.removeLast();
        navigateTo(navHistory.last(), false);  // don't re-add
    } else {
        goHome();
    }
}

void MainWindow::navigateTo(const QString &path, bool addToHistory) {
    abortCurrentScansTemporarily();
    scanAbortFlag = false;
    if (!QDir(path).exists()) return;
    currentPath = path;
    pathLabel->setText(path);
    if (addToHistory)
        navHistory.append(path);
    loadFolder(path);
    stack->setCurrentWidget(folderView);
    loadFaceListFromDatabase();
}

void MainWindow::loadDrives() {
    driveList->clear();
    QFileInfoList drives = QDir::drives();
    QFileIconProvider iconProvider;
    for (const auto& drive : drives) {
        QListWidgetItem *item = new QListWidgetItem(iconProvider.icon(drive), drive.absoluteFilePath());
        item->setData(Qt::UserRole, drive.absoluteFilePath());
        driveList->addItem(item);
    }
}

void MainWindow::loadFolder(const QString &path) {
    scanAbortFlag = false;
    thumbAbortFlag = true;
    QTimer::singleShot(100, this, [this]() {
        thumbAbortFlag = false;
    });

    folderView->clear();
    QDir dir(path);
    QFileInfoList entries = dir.entryInfoList(QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot);
    QFileIconProvider iconProvider;

    std::sort(entries.begin(), entries.end(), [](const QFileInfo& a, const QFileInfo& b) {
        if (a.isDir() != b.isDir()) return a.isDir();
        return a.fileName().toLower() < b.fileName().toLower();
    });

    for (const auto& entry : entries) {
        QListWidgetItem *item = nullptr;

        if (entry.isDir()) {
            QDir folder(entry.absoluteFilePath());
            QFileInfoList files = folder.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);

            bool hasImage = false;
            bool isEmpty = files.isEmpty();

            for (const auto& file : files) {
                QString ext = file.suffix().toLower();
                if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
                    hasImage = true;
                    break;
                }
            }

            if (!hasImage && !isEmpty) continue;  // ⛔ Skip folder with only non-image files

            item = new QListWidgetItem(iconProvider.icon(entry), entry.fileName());
            // ✅ Set initial flags for folders
            item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        }
        else {
            QString ext = entry.suffix().toLower();
            if (!(ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp"))
                continue;

            QString imagePath = entry.absoluteFilePath();
            qDebug() << "📥 Queued image:" << imagePath;

            // Show placeholder first
            QListWidgetItem* item = new QListWidgetItem(QIcon(":/icons/placeholder.png"), entry.fileName());
            item->setToolTip(imagePath);
            item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
            item->setData(Qt::UserRole, imagePath);

            int itemRow = folderView->count();  // Capture position
            folderView->addItem(item);          // Add it BEFORE launching the thread

            QPointer<QListWidget> list = folderView;

            QFuture<void> _ = QtConcurrent::run([this, imagePath, list, itemRow]() {
                qDebug() << "🔄 [Thread] Generating thumbnail for:" << imagePath;

                if (scanAbortFlag) {
                    qDebug() << "⏹️ [Thread] Skipped due to scanAbortFlag:" << imagePath;
                    return;
                }

                QPixmap thumb = getCachedThumbnail(imagePath, QSize(128, 128));

                if (!scanAbortFlag && !thumb.isNull()) {
                    QMetaObject::invokeMethod(list, [=]() {
                        if (scanAbortFlag || !list) {
                            qWarning() << "⚠️ Skipped UI update — scan aborted or list deleted for" << imagePath;
                            return;
                        }

                        if (itemRow < 0 || itemRow >= list->count()) {
                            qWarning() << "⚠️ Skipped UI update — invalid itemRow" << itemRow << "for" << imagePath;
                            return;
                        }

                        QListWidgetItem* item = list->item(itemRow);
                        if (!item) {
                            qWarning() << "⚠️ Skipped UI update — item at row" << itemRow << "was null for" << imagePath;
                            return;
                        }

                        item->setIcon(QIcon(thumb));
                        qDebug() << "✅ [UI] Set thumbnail for:" << imagePath;
                    });

                } else {
                    qWarning() << "❌ [Thread] Failed to generate thumbnail for:" << imagePath;
                }
            });
        }

        if (item && entry.isDir()) {
            item->setToolTip(entry.absoluteFilePath());
            item->setData(Qt::UserRole, entry.absoluteFilePath());
            folderView->addItem(item);
        }

    }
}

void MainWindow::refresh() {
    if (scanAbortFlag) return;
    if (currentPath.isEmpty()) {
        goHome();
        return;
    }

    abortCurrentScansTemporarily();
    navigateTo(currentPath);
    faceList->clear();
    personList.clear();

    statusBar()->showMessage("🔍 Detecting faces in background...", 3000);

    QFuture<void> future = QtConcurrent::run([this]() {
        QDirIterator it(currentPath,
                        QStringList() << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp",
                        QDir::Files,
                        includeSubfolders ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags);

        while (it.hasNext()) {
            QString path = it.next();
            QFileInfo info(path);
            qint64 mtime = info.lastModified().toSecsSinceEpoch();
            if (faceIndexer.faceAlreadyProcessed(path, mtime)) {
                qDebug() << "⏭️ Skipping cached:" << path;
                continue;
            }


            cv::Mat fullRes = cv::imread(path.toStdString());
            if (fullRes.empty()) continue;

            cv::Size resizedSize;
            double scaleX = 1.0, scaleY = 1.0;
            ResizeMode mode = ResizeMode::Original; //Fit1280x720;  // 🔁 or Fit1024x1024 or Original
            cv::Mat matBGR = resizeImageForDetection(fullRes, mode, resizedSize, scaleX, scaleY);

            if (matBGR.empty()) continue;

            auto faces = faceDetector.detectFaces(matBGR);
            qDebug() << "🧠" << faces.size() << "face(s) found in:" << path;

            std::vector<QRect> faceRects;
            std::vector<std::vector<float>> embeddings;
            std::vector<double> symmetries;
            std::vector<double> focuses;

            for (const QRect& rect : faces) {
                auto embedding = faceDetector.getJitteredEmbedding(matBGR, rect);
                if (embedding.empty()) continue;
                //embedding = normalizeEmbedding(embedding);
                qDebug() << "📸 In file:" << path << "📐 Face Size:" << rect.width() << "x" << rect.height();

                if (rect.width() < 20 || rect.height() < 20) continue;
                if (rect.width() > 1000 || rect.height() > 1000) continue;

                dlib::rectangle dlibRect(rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height());
                dlib::cv_image<dlib::bgr_pixel> dlibImg(matBGR);
                auto shape = faceDetector.getLandmarks(dlibImg, dlibRect);
                double symmetry = getSymmetryScore(shape);

                QRect scaled(
                    int(rect.x() * scaleX),
                    int(rect.y() * scaleY),
                    int(rect.width() * scaleX),
                    int(rect.height() * scaleY)
                    );

                cv::Rect roi(scaled.x(), scaled.y(), scaled.width(), scaled.height());
                roi &= cv::Rect(0, 0, fullRes.cols, fullRes.rows);
                cv::Mat faceMat = fullRes(roi).clone();

                cv::resize(faceMat, faceMat, cv::Size(64, 64));
                double focus = getFocusScore(faceMat);
                QImage faceImg(faceMat.data, faceMat.cols, faceMat.rows, faceMat.step, QImage::Format_BGR888);
                QPixmap thumb = QPixmap::fromImage(faceImg.copy());

                bool matched = false;
                for (size_t i = 0; i < personList.size(); ++i) {
                    if (isSimilarFace(embedding, personList[i].embedding, matchDIST)) {
                        matched = true;
                        personList[i].count += 1;

                        if (eyesAreOpen(shape)) {
                            double prevFocus = personList[i].focus;
                            bool focusGood = focus >= goodFocusThreshold;
                            bool focusAcceptable = focus >= (prevFocus - focusTolerance);

                            if (symmetry < personList[i].symmetry && (focusGood || focusAcceptable)) {
                                personList[i].embedding = embedding;
                                personList[i].symmetry = symmetry;
                                personList[i].focus = focus;
                                personList[i].thumb = thumb;
                                personList[i].imagePath = path;

                                if (!scanAbortFlag) {
                                    QMetaObject::invokeMethod(this, [=]() {
                                        if (!scanAbortFlag && faceList && i < faceList->count()) {
                                            faceList->item(static_cast<int>(i))->setIcon(QIcon(thumb));
                                            faceList->item(static_cast<int>(i))->setText(
                                                QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
                                        }
                                    }, Qt::QueuedConnection);
                                }

                            } else {
                                if (!scanAbortFlag) {
                                    QMetaObject::invokeMethod(this, [=]() {
                                        if (!scanAbortFlag && faceList && i < faceList->count()) {
                                            faceList->item(static_cast<int>(i))->setText(
                                                QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
                                        }
                                    }, Qt::QueuedConnection);
                                }

                            }
                        } else {
                            if (!scanAbortFlag) {
                                QMetaObject::invokeMethod(this, [=]() {
                                    if (!scanAbortFlag && faceList && i < faceList->count() && faceList->item(i)) {
                                        faceList->item(static_cast<int>(i))->setText(
                                            QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
                                    }
                                }, Qt::QueuedConnection);
                            }

                        }
                        break;
                    }
                }

                if (!matched) {
                    personList.push_back({embedding, symmetry, focus, thumb, path});
                    if (!scanAbortFlag) {
                        QMetaObject::invokeMethod(this, [=]() {
                            if (!scanAbortFlag && faceList) {
                                QString label = QString("Person %1 (1)").arg(personList.size());
                                QListWidgetItem* item = new QListWidgetItem(QIcon(thumb), label);
                                item->setToolTip(path);
                                item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
                                item->setCheckState(Qt::Unchecked);
                                faceList->addItem(item);
                            }
                        }, Qt::QueuedConnection);
                    }
                }

                faceRects.push_back(rect);
                embeddings.push_back(embedding);
                symmetries.push_back(symmetry);
                focuses.push_back(focus);
            }

            QList<FaceEntry> faceEntries;
            QList<std::vector<float>> embeddingList;

            for (int i = 0; i < static_cast<int>(faceRects.size()); ++i) {
                FaceEntry entry;
                entry.imagePath = path;
                entry.faceRect = faceRects[i];
                entry.quality = focuses[i];
                entry.globalId = "";  // leave empty for now
                faceEntries.append(entry);
                embeddingList.append(embeddings[i]);
            }

            FaceDatabaseManager::instance().addFacesBatch(faceEntries, embeddingList);

        }
        
        if (!scanAbortFlag) {
            QMetaObject::invokeMethod(this, [this]() {
                if (!scanAbortFlag && faceList && statusBar()) {
                    updateFaceList();
                    statusBar()->showMessage(QString("🧠 Faces detected: %1").arg(personList.size()), 2000);
                }
            }, Qt::QueuedConnection);
        }

    });
}


bool MainWindow::isSimilarFace(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
    float dist = l2Distance(a, b);
    bool isMatch = dist < threshold;

    qDebug() << (isMatch ? "✅ Match" : "❌ No Match")
             << " | Distance:" << dist
             << " | Threshold:" << threshold;

    return isMatch;
}


void MainWindow::showFolderViewContextMenu(const QPoint& pos) {
    QMenu menu(this);

    QAction* openAction = menu.addAction("Open");
    QAction* openWithAction = menu.addAction("Open with System App");
    QAction* cutAction = menu.addAction("Cut");

    menu.addSeparator();

    QAction* copyAction = menu.addAction("Copy Selected");
    QAction* pasteAction = menu.addAction("Paste Here");
    QAction* newFolderAction = menu.addAction("New Folder");

    QAction* chosen = menu.exec(folderView->viewport()->mapToGlobal(pos));

    if (chosen == openAction) {
        QListWidgetItem* item = folderView->itemAt(pos);
        if (item) {
            showImagePopup(item->data(Qt::UserRole).toString());
        }
    }

    else if (chosen == openWithAction) {
        QListWidgetItem* item = folderView->itemAt(pos);
        if (!item) return;

        QString path = item->data(Qt::UserRole).toString();
        QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    }

    else if (chosen == copyAction) {
        performCopy();
    }
    else if (chosen == cutAction) {
        performCut();
    }
    else if (chosen == pasteAction) {
        pasteToCurrentFolder();
    }

    else if (chosen == newFolderAction) {
        createNewFolder();
    }

}

void MainWindow::showImagePopup(const QString& path) {
    QFileInfo info(path);
    if (!info.exists() || !info.isFile()) return;

    QImage image(path);
    if (image.isNull()) return;

    QDialog* dialog = new QDialog(this);
    dialog->setWindowTitle(info.fileName());
    dialog->resize(800, 600);

    QLabel* label = new QLabel(dialog);
    label->setPixmap(QPixmap::fromImage(image).scaled(dialog->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    label->setAlignment(Qt::AlignCenter);

    QVBoxLayout* layout = new QVBoxLayout(dialog);
    layout->addWidget(label);
    dialog->setLayout(layout);

    dialog->exec();
}

void MainWindow::pasteToCurrentFolder() {
    if (copiedFilePaths.isEmpty()) {
        statusBar()->showMessage("⚠️ No copied files", 2000);
        return;
    }

    if (currentPath.isEmpty()) {
        statusBar()->showMessage("⚠️ No destination folder selected", 2000);
        return;
    }

    static bool applyToAll = false;
    static bool userChoseOverwrite = false;

    int overwriteCount = 0, skippedCount = 0;

    for (const auto& src : copiedFilePaths) {
        QFileInfo srcInfo(src);
        QString destPath = currentPath + "/" + srcInfo.fileName();

        if (QFile::exists(destPath)) {
            if (!applyToAll) {
                QMessageBox box(this);
                box.setWindowTitle("File Already Exists");
                box.setText(QString("File already exists:\n%1").arg(srcInfo.fileName()));
                box.setInformativeText("Do you want to overwrite it?");
                QAbstractButton* yesBtn = box.addButton("Yes", QMessageBox::YesRole);
                QAbstractButton* noBtn  = box.addButton("No", QMessageBox::NoRole);

                QCheckBox* check = new QCheckBox("Apply to all remaining files");
                box.setCheckBox(check);

                box.exec();

                if (box.clickedButton() == yesBtn) {
                    userChoseOverwrite = true;
                    applyToAll = check->isChecked();
                } else {
                    userChoseOverwrite = false;
                    applyToAll = check->isChecked();
                    ++skippedCount;
                    continue;
                }
            }

            if (!userChoseOverwrite) {
                ++skippedCount;
                continue;
            }

            ++overwriteCount;
        }

        if (QFile::copy(src, destPath)) {
            if (cutMode) {
                QFile srcFile(src);

                // First attempt to remove
                if (!srcFile.remove()) {
                    // Try force permissions and retry
                    srcFile.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
                    if (!srcFile.remove()) {
                        qWarning() << "❌ Failed to remove file after cut:" << src << "Error:" << srcFile.errorString();
                    }
                }
            }
        }
    }

    cutMode = false;

    bool copiedAny = (overwriteCount > 0 || copiedFilePaths.size() > skippedCount);

    statusBar()->showMessage(
        QString("✅ Pasted: %1 file(s), Overwritten: %2, Skipped: %3")
            .arg(copiedFilePaths.size())
            .arg(overwriteCount)
            .arg(skippedCount),
        3000
        );

    if (copiedAny) {
        abortCurrentScansTemporarily();  // ⬅️ Add this
        loadFolder(currentPath);
    }

}

void MainWindow::performCopy() {
    copiedFilePaths.clear();

    // If any face is selected, copy images associated with checked faces
    bool anyFaceChecked = false;
    for (int i = 0; i < faceList->count(); ++i) {
        if (faceList->item(i)->checkState() == Qt::Checked) {
            copiedFilePaths << personList[i].imagePath;
            anyFaceChecked = true;
        }
    }

    // Fallback to folderView selection if no face is checked
    if (!anyFaceChecked) {
        for (auto& item : folderView->selectedItems())
            copiedFilePaths << item->data(Qt::UserRole).toString();
    }

    cutMode = false;
    statusBar()->showMessage(QString("📋 Copied %1 files").arg(copiedFilePaths.size()), 2000);
}

void MainWindow::performCut() {
    copiedFilePaths.clear();
    for (auto& item : folderView->selectedItems())
        copiedFilePaths << item->data(Qt::UserRole).toString();
    cutMode = true;
    statusBar()->showMessage(QString("✂️ Cut %1 files").arg(copiedFilePaths.size()), 2000);
}

void MainWindow::performPaste() {
    pasteToCurrentFolder();
}

void MainWindow::keyPressEvent(QKeyEvent* event) {
    if (event->modifiers() & Qt::ControlModifier) {
        switch (event->key()) {
        case Qt::Key_C:  // Ctrl+C
            performCopy();
            break;

        case Qt::Key_X:  // Ctrl+X
            performCut();
            break;

        case Qt::Key_V:  // Ctrl+V
            performPaste();
            break;

        case Qt::Key_A:  // Ctrl+A
            folderView->selectAll();
            break;

        case Qt::Key_Z:  // Ctrl+Z — undo paste (just reload folder for now)
            statusBar()->showMessage("↩️ Undo not supported, refreshing view.", 2000);
            loadFolder(currentPath);
            break;
        }
    } else if (event->key() == Qt::Key_Delete) {
        QList<QListWidgetItem*> selectedItems = folderView->selectedItems();
        if (selectedItems.isEmpty()) return;

        int confirm = QMessageBox::question(this, "Delete Files",
                                            QString("Are you sure you want to delete %1 file(s)?")
                                                .arg(selectedItems.size()),
                                            QMessageBox::Yes | QMessageBox::No);

        if (confirm == QMessageBox::Yes) {
            for (auto& item : selectedItems) {
                QString path = item->data(Qt::UserRole).toString();
                QFile::remove(path);
            }
            loadFolder(currentPath);
            statusBar()->showMessage("🗑️ Deleted selected files", 2000);
        }
    }
    else if (event->key() == Qt::Key_Backspace) {
        goBack();
        return;    // prevent base class from interfering
    }


    QMainWindow::keyPressEvent(event);  // base call
}

void MainWindow::updateFaceList() {
    std::sort(personList.begin(), personList.end(), [](const FaceStats& a, const FaceStats& b) {
        return a.count > b.count;
    });

    faceList->clear();

    for (int i = 0; i < static_cast<int>(personList.size()); ++i) {
        const auto& p = personList[i];

        // Filter by includeSubfolders setting
        QFileInfo imgInfo(p.imagePath);
        QString parentPath = imgInfo.absolutePath();

        bool inCurrentFolder = (parentPath == currentPath);
        bool inSubfolder = parentPath.startsWith(currentPath + QDir::separator());

        if (!includeSubfolders && !inCurrentFolder)
            continue;

        QString label = QString("Person %1 (%2)").arg(i + 1).arg(p.count);
        QListWidgetItem* item = new QListWidgetItem(QIcon(p.thumb), label);
        item->setToolTip(p.imagePath);
        item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Unchecked);
        faceList->addItem(item);
    }
}

void MainWindow::updateFolderViewThumbnails(const QString& folder) {
    for (int i = 0; i < folderView->count(); ++i) {
        QListWidgetItem* item = folderView->item(i);
        QString imagePath = item->data(Qt::UserRole).toString();
        QFileInfo info(imagePath);

        // Only apply to files (not folders) inside the target folder
        if (!info.isFile() || info.absolutePath() != folder)
            continue;

        // Run thumbnail loading in background
        QFuture<void> future = QtConcurrent::run([=]() {
            QPixmap thumb = getCachedThumbnail(imagePath, QSize(128, 128));
            if (!scanAbortFlag && !thumb.isNull()) {
                QMetaObject::invokeMethod(folderView, [=]() {
                    if (!scanAbortFlag && folderView && item && folderView->indexFromItem(item).isValid()) {
                        item->setIcon(QIcon(thumb));
                    }
                }, Qt::QueuedConnection);
            }

        });
    }
}

void MainWindow::updateFolderViewCheckboxesFromFaceSelection() {
    QVector<std::vector<float>> selectedEmbeddings;

    for (int i = 0; i < faceList->count(); ++i) {
        if (faceList->item(i)->checkState() == Qt::Checked) {
            selectedEmbeddings.push_back(personList[i].embedding);
        }
    }

    for (int i = 0; i < folderView->count(); ++i) {
        QListWidgetItem* item = folderView->item(i);
        QString path = item->data(Qt::UserRole).toString();
        QFileInfo info(path);

        if (!info.isFile()) continue;

        // Always allow user to manually check/uncheck
        item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Unchecked);  // default to unchecked

        // Skip matching logic if no face selected
        if (selectedEmbeddings.isEmpty()) continue;

        QList<FaceEntry> faces = includeSubfolders
                                     ? FaceDatabaseManager::instance().getFaceEntriesInSubtree(currentPath)
                                     : FaceDatabaseManager::instance().getFaceEntriesInFolder(info.absolutePath());

        for (const FaceEntry& face : faces) {
            if (QFileInfo(face.imagePath).fileName() != info.fileName())
                continue;

            std::vector<float> emb = FaceDatabaseManager::instance().getEmbeddingById(face.id);
            for (const auto& selected : selectedEmbeddings) {
                if (isSimilarFace(selected, emb, matchDIST)) {
                    item->setCheckState(Qt::Checked);
                    goto matched;
                }
            }
        }
    matched:;
    }
}

void MainWindow::loadFaceListFromDatabase() {
    personList.clear();

    QList<FaceEntry> entries;
    if (includeSubfolders) {
        entries = FaceDatabaseManager::instance().getFaceEntriesInSubtree(currentPath);
    } else {
        entries = FaceDatabaseManager::instance().getFaceEntriesInFolder(currentPath);
    }

    for (const auto& face : entries) {
        std::vector<float> emb = FaceDatabaseManager::instance().getEmbeddingById(face.id);
        if (emb.empty()) continue;

        QImage image(face.imagePath);
        if (image.isNull()) continue;

        QRect r = face.faceRect;
        QImage faceImage = image.copy(r).scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        QPixmap thumb = QPixmap::fromImage(faceImage);

        bool matched = false;
        for (auto& p : personList) {
            if (isSimilarFace(p.embedding, emb, matchDIST)) {
                p.count++;
                matched = true;
                break;
            }
        }

        if (!matched) {
            personList.push_back({emb, 0.0, 0.0, thumb, face.imagePath});
        }
    }

    updateFaceList();
}

void MainWindow::createNewFolder() {
    if (currentPath.isEmpty()) {
        statusBar()->showMessage("⚠️ No destination folder selected.", 2000);
        return;
    }

    bool ok;
    QString folderName = QInputDialog::getText(this, "Create New Folder",
                                               "Enter folder name:", QLineEdit::Normal,
                                               "New Folder", &ok);
    if (!ok || folderName.trimmed().isEmpty()) return;

    QString fullPath = QDir::cleanPath(currentPath + "/" + folderName);
    if (QDir(fullPath).exists()) {
        QMessageBox::warning(this, "Already Exists", "A folder with that name already exists.");
    } else if (QDir().mkpath(fullPath)) {
        statusBar()->showMessage("✅ Created folder: " + folderName, 2000);
    } else {
        QMessageBox::warning(this, "Error", "⚠️ Could not create folder.");
    }

    QTimer::singleShot(100, this, [this]() {
        loadFolder(currentPath);
    });
}

void MainWindow::abortCurrentScansTemporarily() {
    scanAbortFlag = true;
    thumbAbortFlag = true;
    qDebug() << "⚠️ Aborting scans temporarily";

    QTimer::singleShot(100, this, [this]() {
        scanAbortFlag = false;
        thumbAbortFlag = false;
        qDebug() << "✅ Resuming scan operations";
    });
}

