#ifndef FACELISTITEMDELEGATE_H
#define FACELISTITEMDELEGATE_H

#include <QStyledItemDelegate>
#include <QApplication>
#include <QPainter>
#include <QStyleOptionViewItem>
#include <QMouseEvent>
#include <QEvent>

class FaceListItemDelegate : public QStyledItemDelegate {
    Q_OBJECT
public:
    enum ViewMode {
        FaceListMode,   // Always show checkbox, center image
        FolderViewMode  // Only show checkbox if item is user-checkable
    };

    explicit FaceListItemDelegate(ViewMode mode, QObject* parent = nullptr);

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    bool editorEvent(QEvent* event, QAbstractItemModel* model,
                     const QStyleOptionViewItem& option, const QModelIndex& index) override;

private:
    ViewMode mode;
};

#endif // FACELISTITEMDELEGATE_H
