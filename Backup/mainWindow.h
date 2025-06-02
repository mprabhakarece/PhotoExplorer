#pragma once

#include <QMainWindow>
#include <QStackedWidget>
#include <QListWidget>
#include <QLabel>
#include <QStringList>

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
    QLabel *pathLabel;
    QString currentPath;
    QStringList navHistory;

    void navigateTo(const QString &path);
    void loadDrives();
    void loadFolder(const QString &path);
};
