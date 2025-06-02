#pragma once

#include <QMainWindow>
#include <QStackedWidget>
#include <QListWidget>
#include <QLabel>
#include <QStringList>
#include "faceindexer.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void goBack();
    void goHome();
    void refresh();

private:
    QStackedWidget *stack;
    QListWidget *driveList;
    QListWidget *folderView;
    QListWidget *faceList;
    QLabel *pathLabel;
    QString currentPath;
    QStringList navHistory;
    bool includeSubfolders = false;
    FaceIndexer* faceIndexer = nullptr;

    void navigateTo(const QString &path);
    void loadDrives();
    void loadFolder(const QString &path);
    QIcon generateFolderPreviewIcon(const QString &folderPath, QSize thumbSize);

};
