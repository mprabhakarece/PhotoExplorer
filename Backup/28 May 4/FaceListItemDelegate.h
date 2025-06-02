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
    explicit FaceListItemDelegate(QObject* parent = nullptr);

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    bool editorEvent(QEvent* event, QAbstractItemModel* model,
                     const QStyleOptionViewItem& option, const QModelIndex& index) override;
};

#endif // FACELISTITEMDELEGATE_H
