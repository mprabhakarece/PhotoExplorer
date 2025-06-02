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

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "mainwindow.h"
#include "faceindexer.h"
#include "FaceListItemDelegate.h"

FaceIndexer faceIndexer;


constexpr double matchDIST = 0.5f;

constexpr double goodFocusThreshold = 100.0; // e.g. ideal Laplacian variance
constexpr double focusTolerance = 25.0;      // how far below is still acceptable

cv::Mat resizeToFixedHeight(const cv::Mat& input, cv::Size& newSize, double& scaleX, double& scaleY) {
    const int targetHeight = 768;
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
    QDir cacheDir(info.dir().absolutePath() + "/.cache/thumbnails");
    if (!cacheDir.exists()) cacheDir.mkpath(".");

    QString thumbPath = cacheDir.absoluteFilePath(info.fileName() + ".thumb");
    if (QFile::exists(thumbPath)) {
        QPixmap cached(thumbPath);
        if (!cached.isNull()) return cached;
    }

    // ✅ Use QImageReader to auto-handle EXIF orientation
    QImageReader reader(imagePath);
    reader.setAutoTransform(true);  // <-- this is the key
    QImage image = reader.read();

    if (image.isNull()) return QPixmap();  // Fallback

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
        statusBar()->showMessage("Include Subfolders " + QString(checked ? "enabled" : "disabled") + ". Click Refresh to apply.", 3000);
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
    faceList->setIconSize(QSize(84, 84));
    faceList->setGridSize(QSize(90, 110)); // icon + label
    faceList->setSpacing(10);
    faceList->setResizeMode(QListView::Adjust);
    faceList->setMovement(QListView::Static);
    faceList->setMaximumWidth(200);
    faceList->setItemDelegate(new FaceListItemDelegate(this));

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
    folderView->setSelectionMode(QAbstractItemView::SingleSelection);
    folderView->setFocusPolicy(Qt::StrongFocus);
    folderView->setFocus();
    folderView->setItemDelegate(new FaceListItemDelegate(this));
    stack->addWidget(folderView);

    connect(folderView, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *item) {
        QFileInfo info(item->data(Qt::UserRole).toString());
        if (info.isDir()) {
            navigateTo(info.absoluteFilePath());
        }
    });

    connect(faceList, &QListWidget::itemChanged, this, [this](QListWidgetItem *item) {
        if (item->checkState() == Qt::Checked) {
            int index = faceList->row(item);
            if (index >= 0 && index < static_cast<int>(personList.size())) {
                const QString selectedFacePath = personList[index].imagePath;
                const auto& selectedEmbedding = personList[index].embedding;

                for (int i = 0; i < folderView->count(); ++i) {
                    QListWidgetItem* imgItem = folderView->item(i);
                    const QString imagePath = imgItem->data(Qt::UserRole).toString();
                    QJsonArray log = faceIndexer.getFaceLog(QFileInfo(imagePath).absolutePath());

                    for (const QJsonValue& val : log) {
                        QJsonObject entry = val.toObject();
                        if (entry["image"].toString() != QFileInfo(imagePath).fileName())
                            continue;

                        QJsonArray faces = entry["faces"].toArray();
                        for (const QJsonValue& faceVal : faces) {
                            QJsonObject faceObj = faceVal.toObject();
                            QJsonArray embArray = faceObj["embedding"].toArray();
                            std::vector<float> emb;
                            for (const auto& v : embArray)
                                emb.push_back(v.toDouble());

                            if (isSimilarFace(selectedEmbedding, emb, matchDIST)) {
                                imgItem->setFlags(imgItem->flags() | Qt::ItemIsUserCheckable);
                                imgItem->setCheckState(Qt::Checked);
                                break;
                            }
                        }
                    }
                }
            }
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
            item->setToolTip(entry.absoluteFilePath());
            item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
            item->setCheckState(Qt::Unchecked);

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

    navigateTo(currentPath);
    faceList->clear();
    personList.clear();

    statusBar()->showMessage("🔍 Detecting faces in background...", 3000);

    QtConcurrent::run([this]() {
        QDirIterator it(currentPath,
                        QStringList() << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp",
                        QDir::Files,
                        includeSubfolders ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags);

        while (it.hasNext()) {
            QString path = it.next();
            QString folder = QFileInfo(path).absolutePath();
            QJsonArray folderLog = faceIndexer.getFaceLog(folder);

            if (faceIndexer.hasFaceData(path, folderLog)) {
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
                if (rect.width() < 40 || rect.height() < 40) continue;
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

                                QMetaObject::invokeMethod(this, [=]() {
                                    faceList->item(static_cast<int>(i))->setIcon(QIcon(thumb));
                                    faceList->item(static_cast<int>(i))->setText(QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
                                }, Qt::QueuedConnection);
                            } else {
                                QMetaObject::invokeMethod(this, [=]() {
                                    faceList->item(static_cast<int>(i))->setText(QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
                                }, Qt::QueuedConnection);
                            }
                        } else {
                            QMetaObject::invokeMethod(this, [=]() {
                                faceList->item(static_cast<int>(i))->setText(QString("Person %1 (%2)").arg(i + 1).arg(personList[i].count));
                            }, Qt::QueuedConnection);
                        }
                        break;
                    }
                }

                if (!matched) {
                    personList.push_back({embedding, symmetry, focus, thumb, path});
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

            faceIndexer.addFaceData(path, faceRects, embeddings, symmetries, focuses);
        }

        QMetaObject::invokeMethod(this, [this]() {
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

            connect(faceList, &QListWidget::itemChanged, this, [this](QListWidgetItem* item) {
                if (item->checkState() == Qt::Checked) {
                    int index = faceList->row(item);
                    if (index >= 0 && index < static_cast<int>(personList.size())) {
                        const auto& selected = personList[index];

                        for (int i = 0; i < folderView->count(); ++i) {
                            QListWidgetItem* imgItem = folderView->item(i);
                            const QString imagePath = imgItem->data(Qt::UserRole).toString();
                            QJsonArray log = faceIndexer.getFaceLog(QFileInfo(imagePath).absolutePath());

                            for (const QJsonValue& val : log) {
                                QJsonObject entry = val.toObject();
                                if (entry["image"].toString() != QFileInfo(imagePath).fileName()) continue;

                                QJsonArray faces = entry["faces"].toArray();
                                for (const QJsonValue& faceVal : faces) {
                                    QJsonObject faceObj = faceVal.toObject();
                                    QJsonArray embArray = faceObj["embedding"].toArray();
                                    std::vector<float> emb;
                                    for (const auto& v : embArray)
                                        emb.push_back(v.toDouble());

                                    if (isSimilarFace(selected.embedding, emb, matchDIST)) {
                                        imgItem->setFlags(imgItem->flags() | Qt::ItemIsUserCheckable);
                                        imgItem->setCheckState(Qt::Checked);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            });

            statusBar()->showMessage(QString("🧠 Faces detected: %1").arg(personList.size()), 2000);
        }, Qt::QueuedConnection);
    });
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


