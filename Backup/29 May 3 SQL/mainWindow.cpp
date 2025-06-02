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

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mainwindow.h"
#include "faceindexer.h"
#include "FaceListItemDelegate.h"
#include "scanworker.h"
#include "FaceDatabaseManager.h"

QQueue<ScanRequest> scanQueue;
QMutex scanQueueMutex;
QWaitCondition scanQueueNotEmpty;
bool scanWorkerRunning = false;


FaceIndexer faceIndexer;


constexpr double matchDIST = 0.4f;

constexpr double goodFocusThreshold = 100.0; // e.g. ideal Laplacian variance
constexpr double focusTolerance = 25.0;      // how far below is still acceptable

std::vector<float> normalizeEmbedding(const std::vector<float>& emb) {
    float norm = std::sqrt(std::inner_product(emb.begin(), emb.end(), emb.begin(), 0.0f));
    std::vector<float> result(emb.size());
    if (norm > 0.0f) {
        std::transform(emb.begin(), emb.end(), result.begin(), [=](float v) { return v / norm; });
    }
    return result;
}

cv::Mat resizeToFixedHeight(const cv::Mat& input, cv::Size& newSize, double& scaleX, double& scaleY) {
    const int targetHeight = 768 * 2;
    double ratio = static_cast<double>(targetHeight) / input.rows;
    int targetWidth = static_cast<int>(input.cols * ratio);

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_AREA);

    scaleX = static_cast<double>(input.cols) / targetWidth;
    scaleY = static_cast<double>(input.rows) / targetHeight;
    newSize = cv::Size(targetWidth, targetHeight);
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

    // Root cache directory: where the executable is located
    QString exeDir = QCoreApplication::applicationDirPath();
    QString absolutePath = info.absolutePath();
    QString driveMapped = absolutePath;

    // Replace ':' in drive letters for safe folder name
    driveMapped.replace(':', '_');

    // Full cache path: .cache/<original_path_structure>/
    QDir cacheDir(exeDir + "/.cache/" + driveMapped);
    if (!cacheDir.exists())
        cacheDir.mkpath(".");

    QString thumbPath = cacheDir.absoluteFilePath(info.fileName() + ".thumb");
    if (QFile::exists(thumbPath)) {
        QPixmap cached(thumbPath);
        if (!cached.isNull()) return cached;
    }

    // ✅ Use QImageReader to auto-handle EXIF orientation
    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    QImage image = reader.read();
    if (image.isNull()) return QPixmap();

    QPixmap thumb = QPixmap::fromImage(
        image.scaled(size, Qt::KeepAspectRatio, Qt::SmoothTransformation)
        );

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
    QPushButton *photScanButton = new QPushButton("Photo Scan");
    QCheckBox* showHiddenFoldersCheckbox = new QCheckBox("Show Hidden Folders");

    QCheckBox *includeSubfoldersCheckbox = new QCheckBox("Include Subfolders");

    QPushButton *copyButton = new QPushButton("Copy Selected");
    QPushButton *pasteButton = new QPushButton("Paste Here");
    QPushButton *createFolderButton = new QPushButton("New Folder");

    pathLabel = new QLabel("This PC");

    toolbarLayout->addWidget(backButton);
    toolbarLayout->addWidget(homeButton);
    toolbarLayout->addWidget(pathLabel);
    toolbarLayout->addWidget(showHiddenFoldersCheckbox);

    toolbarLayout->addStretch();

    toolbarLayout->addWidget(createFolderButton);
    toolbarLayout->addWidget(copyButton);
    toolbarLayout->addWidget(pasteButton);
    toolbarLayout->addWidget(includeSubfoldersCheckbox);
    toolbarLayout->addWidget(photScanButton);

    toolbar->addWidget(toolbarWidget);
    toolbarWidget->setMinimumHeight(36);

    connect(backButton, &QPushButton::clicked, this, &MainWindow::goBack);
    connect(homeButton, &QPushButton::clicked, this, &MainWindow::goHome);
    connect(includeSubfoldersCheckbox, &QCheckBox::toggled, this, [=](bool checked) {
        includeSubfolders = checked;
        statusBar()->showMessage("Include Subfolders " + QString(checked ? "enabled" : "disabled") + ". Click Refresh to apply.", 3000);
    });

    connect(photScanButton, &QPushButton::clicked, this, &MainWindow::refresh);

    QSplitter *mainSplitter = new QSplitter(this);
    setCentralWidget(mainSplitter);

    faceList = new QListWidget(this);
    faceList->setViewMode(QListView::IconMode);
    faceList->setIconSize(QSize(84, 84));
    faceList->setGridSize(QSize(90, 110)); // icon + label
    faceList->setSpacing(10);
    faceList->setResizeMode(QListView::Adjust);
    faceList->setMovement(QListView::Static);
    faceList->setMaximumWidth(200);
    faceList->setItemDelegate(new FaceListItemDelegate(FaceListItemDelegate::FaceListMode, this));

    //faceList->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    //folderView->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

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
    folderView->setGridSize(QSize(140, 172)); // 128 icon + 16 label + 8 spacing + safety
    folderView->setSpacing(0);                // no gap between cells
    folderView->setResizeMode(QListView::Adjust);
    folderView->setMovement(QListView::Static);
    folderView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    folderView->setFocusPolicy(Qt::StrongFocus);
    folderView->setFocus();
    folderView->setItemDelegate(new FaceListItemDelegate(FaceListItemDelegate::FolderViewMode, this));
    stack->addWidget(folderView);

    // Right-click menu support
    folderView->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(folderView, &QListWidget::customContextMenuRequested,
            this, &MainWindow::showFolderViewContextMenu);

    connect(folderView, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
        QString path = item->data(Qt::UserRole).toString();
        QFileInfo info(path);

        if (info.isDir()) {
            navigateTo(info.absoluteFilePath());  // ✅ safely open folder
        } else if (info.isFile()) {
            showImagePopup(path);  // ✅ open image only if it's valid
        }
    });

    connect(faceList, &QListWidget::itemChanged,
            this, &MainWindow::updateFolderViewCheckboxesFromFaceSelection);


    connect(copyButton, &QPushButton::clicked, this, [this]() {
        performCopy();
    });

    connect(pasteButton, &QPushButton::clicked, this, [this]() {
        performPaste();
    });

    connect(createFolderButton, &QPushButton::clicked, this, [this]() {
        if (currentPath.isEmpty()) {
            statusBar()->showMessage("⚠️ No destination folder selected.", 2000);
            return;
        }

        bool ok;
        QString folderName = QInputDialog::getText(this, "Create New Folder",
                                                   "Enter folder name:",
                                                   QLineEdit::Normal,
                                                   "New Folder", &ok);
        if (!ok || folderName.trimmed().isEmpty()) return;

        QString fullPath = currentPath + "/" + folderName;
        if (QDir(fullPath).exists()) {
            statusBar()->showMessage("⚠️ Folder already exists: " + folderName, 2000);
            return;
        }

        if (!QDir(fullPath).exists()) {
            QDir().mkpath(fullPath);
            QTimer::singleShot(100, this, [this]() {
                loadFolder(currentPath);
            });
        }

        else {
            statusBar()->showMessage("❌ Failed to create folder", 2000);
        }
    });

    connect(showHiddenFoldersCheckbox, &QCheckBox::toggled, this, [=](bool checked) {
        showHiddenFolders = checked;
        loadFolder(currentPath);
    });



    QFuture<void> _ = QtConcurrent::run([this]() {
        while (true) {
            scanQueueMutex.lock();
            while (scanQueue.isEmpty()) {
                scanQueueNotEmpty.wait(&scanQueueMutex);  // wait for work
            }
            ScanRequest req = scanQueue.dequeue();
            scanQueueMutex.unlock();

            QMetaObject::invokeMethod(this, [this]() {
                statusBar()->showMessage("🔄 Photo scan started...");
            });

            //performScan(req.folderPath, req.includeSubfolders);
            //addFaceToPersonList(embedding, thumb, path, shape, focus, symmetry);

            QMetaObject::invokeMethod(this, [this]() {
                statusBar()->showMessage("✅ Photo scan finished.");
            });
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
        navigateTo(navHistory.last(), false);  // don't re-add
    } else {
        goHome();
    }
}

void MainWindow::navigateTo(const QString &path, bool addToHistory) {
    if (!QDir(path).exists()) return;
    currentPath = path;
    pathLabel->setText(path);
    if (addToHistory)
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
            QDir folder(entry.absoluteFilePath());
            QFileInfoList files = folder.entryInfoList(QDir::Files | QDir::NoDotAndDotDot);

            bool hasImage = false;
            bool isEmpty = files.isEmpty();

            for (const QFileInfo& file : files) {
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

            // Show placeholder first, then load thumbnail async
            item = new QListWidgetItem(QIcon(":/icons/placeholder.png"), entry.fileName());
            item->setToolTip(entry.absoluteFilePath());
            // ✅ Set initial flags without checkbox
            item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);

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

void MainWindow::performScan(const QString& folder, bool recursive) {
    QMetaObject::invokeMethod(this, [=]() {
        navigateTo(folder);  // refresh view
        faceList->clear();
        personList.clear();
    }, Qt::QueuedConnection);

    QDirIterator it(folder,
                    QStringList() << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp",
                    QDir::Files,
                    recursive ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags);

    while (it.hasNext()) {
        QString path = it.next();
        QString imgFolder = QFileInfo(path).absolutePath();

        QList<FaceEntry> entries = faceIndexer.getFaceEntriesInFolder(imgFolder);

        if (faceIndexer.faceAlreadyProcessed(imgFolder, QFileInfo(imgFolder).lastModified().toSecsSinceEpoch())) {
            qDebug() << "⏭️ Skipping cached:" << path;
            continue;
        }

        cv::Mat fullRes = cv::imread(path.toStdString());
        if (fullRes.empty()) continue;

        double scaleX = 1.0, scaleY = 1.0;
        cv::Size resizedSize;
        cv::Mat matBGR = resizeToFixedHeight(fullRes, resizedSize, scaleX, scaleY);
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
            embedding = normalizeEmbedding(embedding);

            if (rect.width() < 20 || rect.height() < 20 || rect.width() > 1000 || rect.height() > 1000)
                continue;

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
                    qDebug() << "✅ Match found with Person" << i << "| Count:" << personList[i].count;
                    qDebug() << "→ Eyes open?" << eyesAreOpen(shape);
                    qDebug() << "→ New symmetry:" << symmetry << "| Prev symmetry:" << personList[i].symmetry;
                    qDebug() << "→ New focus:" << focus << "| Prev focus:" << personList[i].focus;

                    if (eyesAreOpen(shape)) {
                        double prevFocus = personList[i].focus;
                        bool focusGood = focus >= goodFocusThreshold;
                        bool focusAcceptable = focus >= (prevFocus - focusTolerance);

                        if (symmetry < personList[i].symmetry && (focusGood || focusAcceptable)) {
                            qDebug() << "✅ Updating stored face for Person" << i;

                            personList[i].embedding = embedding;
                            personList[i].symmetry = symmetry;
                            personList[i].focus = focus;
                            personList[i].thumb = thumb;
                            personList[i].imagePath = path;
                        } else {
                            qDebug() << "⚠️ Skipped update: not enough quality improvement.";
                        }
                    } else {
                        qDebug() << "⚠️ Skipped update: eyes not open.";
                    }

                    QMetaObject::invokeMethod(this, [=]() {
                        faceList->item(static_cast<int>(i))->setIcon(QIcon(personList[i].thumb));
                        faceList->item(static_cast<int>(i))->setText(QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
                    }, Qt::QueuedConnection);

                    break;
                }

            }

            if (!matched) {
                FaceStats stats = {embedding, symmetry, focus, thumb, path};
                personList.push_back(stats);

                QImage safeImage = faceImg.copy();
                QPixmap thumb = QPixmap::fromImage(safeImage);
                QIcon icon(thumb);
                QMetaObject::invokeMethod(this, [=]() {
                    QString label = QString("Person %1 (1)").arg(personList.size());
                    QListWidgetItem* item = new QListWidgetItem(QIcon(thumb), label);
                    item->setToolTip(path);
                    item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
                    item->setCheckState(Qt::Unchecked);
                    faceList->addItem(item);
                }, Qt::QueuedConnection);
            }

            faceRects.push_back(rect);
            embeddings.push_back(embedding);
            symmetries.push_back(symmetry);
            focuses.push_back(focus);
        }

        for (size_t i = 0; i < faceRects.size(); ++i) {
            faceIndexer.saveFaceEntry(path, faceRects[i], embeddings[i], focuses[i],
                                      QFileInfo(path).lastModified().toSecsSinceEpoch());
        }


    }

    QMetaObject::invokeMethod(this, [this]() {
        updateFaceList();
    }, Qt::QueuedConnection);
}

void MainWindow::refresh() {
    if (currentPath.isEmpty()) {
        goHome();
        return;
    }

    QMutexLocker locker(&scanQueueMutex);
    scanQueue.enqueue({ currentPath, includeSubfolders });
    scanQueueNotEmpty.wakeOne();  // Notify worker
}


bool MainWindow::isSimilarFace(const std::vector<float>& a, const std::vector<float>& b, float threshold) {
    if (a.size() != b.size()) {
        qDebug() << "❌ Embedding size mismatch:" << a.size() << "vs" << b.size();
        return false;
    }

    float distSq = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        distSq += diff * diff;
    }

    float dist = std::sqrt(distSq);
    bool isMatch = dist < threshold;

    qDebug() << (isMatch ? "✅ Match" : "❌ No Match")
             << " | Distance:" << dist
             << " | Threshold:" << threshold;

    return isMatch;
}

void MainWindow::addFaceToPersonList(
    const std::vector<float>& embedding,
    const QPixmap& thumb,
    const QString& path,
    const dlib::full_object_detection& shape,
    double focus,
    double symmetry)
{
    bool matched = false;
    for (size_t i = 0; i < personList.size(); ++i) {
        if (isSimilarFace(embedding, personList[i].embedding, matchDIST)) {
            matched = true;
            personList[i].count += 1;
            qDebug() << "✅ Match found with Person" << i << "| Count:" << personList[i].count;
            qDebug() << "→ Eyes open?" << eyesAreOpen(shape);
            qDebug() << "→ New symmetry:" << symmetry << "| Prev symmetry:" << personList[i].symmetry;
            qDebug() << "→ New focus:" << focus << "| Prev focus:" << personList[i].focus;

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
                }
            }

            QMetaObject::invokeMethod(this, [=]() {
                faceList->item(static_cast<int>(i))->setIcon(QIcon(personList[i].thumb));
                faceList->item(static_cast<int>(i))->setText(QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
            }, Qt::QueuedConnection);

            return;
        }
    }

    // Add as new person
    FaceStats stats = {embedding, symmetry, focus, thumb, path};
    personList.push_back(stats);

    QMetaObject::invokeMethod(this, [=]() {
        QString label = QString("Person %1 (1)").arg(personList.size());
        QListWidgetItem* item = new QListWidgetItem(QIcon(thumb), label);
        item->setToolTip(path);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Unchecked);
        faceList->addItem(item);
    }, Qt::QueuedConnection);
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
        bool ok;
        QString folderName = QInputDialog::getText(this, "Create New Folder",
                                                   "Enter folder name:",
                                                   QLineEdit::Normal,
                                                   "New Folder", &ok);
        if (ok && !folderName.trimmed().isEmpty()) {
            QString fullPath = currentPath + "/" + folderName;
            if (QDir().mkpath(fullPath)) {
                statusBar()->showMessage("✅ Created folder: " + folderName, 2000);
                QTimer::singleShot(100, this, [this]() {
                    loadFolder(currentPath);
                });
                QDir().refresh();
            }
             else {
                QMessageBox::warning(this, "Already Exists", "A folder with that name already exists.");
            }
        }
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

    for (const QString& src : copiedFilePaths) {
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

    if (copiedAny)
        loadFolder(currentPath);
}

void MainWindow::performCopy() {
    copiedFilePaths.clear();
    for (QListWidgetItem* item : folderView->selectedItems())
        copiedFilePaths << item->data(Qt::UserRole).toString();
    cutMode = false;
    statusBar()->showMessage(QString("📋 Copied %1 files").arg(copiedFilePaths.size()), 2000);
}

void MainWindow::performCut() {
    copiedFilePaths.clear();
    for (QListWidgetItem* item : folderView->selectedItems())
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
            for (QListWidgetItem* item : selectedItems) {
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
        QFuture<void> _ = QtConcurrent::run([=]() {
            QPixmap thumb = getCachedThumbnail(imagePath, QSize(128, 128));
            if (!thumb.isNull()) {
                QMetaObject::invokeMethod(folderView, [=]() {
                    item->setIcon(QIcon(thumb));
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

    // Reset all items
    for (int i = 0; i < folderView->count(); ++i) {
        QListWidgetItem* imgItem = folderView->item(i);
        if (QFileInfo(imgItem->data(Qt::UserRole).toString()).isFile()) {
            imgItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
            imgItem->setCheckState(Qt::Unchecked);
        }
    }

    if (selectedEmbeddings.isEmpty()) return;

    for (int i = 0; i < folderView->count(); ++i) {
        QListWidgetItem* imgItem = folderView->item(i);
        const QString path = imgItem->data(Qt::UserRole).toString();
        QFileInfo info(path);
        if (info.isDir()) continue;

        QList<FaceEntry> entries = faceIndexer.getFaceEntriesInFolder(info.absolutePath());

        for (const FaceEntry& entry : entries) {
            if (entry.imagePath != path) continue;

            std::vector<float> emb = FaceDatabaseManager::instance().getEmbeddingById(entry.id);

            for (const auto& selected : selectedEmbeddings) {
                if (isSimilarFace(selected, emb, matchDIST)) {
                    imgItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
                    imgItem->setCheckState(Qt::Checked);
                    goto matched;
                }
            }
        }

    matched:;
    }
}

