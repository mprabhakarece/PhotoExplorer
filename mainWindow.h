#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QListWidget>
#include <QStackedWidget>
#include <QLabel>
#include <QStringList>
#include "faceDetector.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    QLabel *pathLabel;
    QListWidget *faceList;
    QListWidget *driveList;
    QListWidget *folderView;
    QStackedWidget *stack;

    QString currentPath;
    QStringList navHistory;
    bool includeSubfolders = false;

    FaceDetector faceDetector;
    std::vector<std::vector<float>> knownEmbeddings;
    QStringList knownFaceThumbs;

    QStringList copiedFilePaths;
    bool showHiddenFolders = false;

    bool cutMode = false;
    void pasteToCurrentFolder();
    void performCopy();
    void performCut();
    void performPaste();
    void showFolderViewContextMenu(const QPoint& pos);
    void showImagePopup(const QString& path);

    void loadDrives();
    void loadFolder(const QString &path);
    void navigateTo(const QString &path, bool addToHistory = true);
    bool isSimilarFace(const std::vector<float>& a, const std::vector<float>& b, float threshold = 0.6f); // ✅ stays here

private slots:
    void goHome();
    void goBack();
    void refresh();
    void updateFaceList();
    void updateFolderViewThumbnails(const QString& folder);
    void updateFolderViewCheckboxesFromFaceSelection();


protected:
    void keyPressEvent(QKeyEvent* event) override;

};

#endif // MAINWINDOW_H
