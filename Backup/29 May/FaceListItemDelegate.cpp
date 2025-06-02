#include "FaceListItemDelegate.h"
#include <QStyle>
#include <QFileInfo>
#include <QPainter>
#include <QApplication>
#include <QMouseEvent>

FaceListItemDelegate::FaceListItemDelegate(QObject* parent)
    : QStyledItemDelegate(parent) {}

QSize FaceListItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const {
    Q_UNUSED(option);
    Q_UNUSED(index);
    return QSize(140, 160);  // Match your grid size
}

void FaceListItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const {
    painter->save();
    QRect itemRect = option.rect;

    // === 1. Draw selection border
    if (option.state & QStyle::State_Selected) {
        QPen pen(QColor(100, 150, 255), 2);  // Light blue border
        painter->setPen(pen);
        painter->drawRect(itemRect.adjusted(1, 1, -2, -2));
    }

    // === 2. Get icon and compute actual scaled draw rect
    QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
    QSize iconSize = option.decorationSize;
    QPixmap pixmap = icon.pixmap(iconSize);  // Actual icon pixmap
    QSize actualSize = pixmap.size().scaled(iconSize, Qt::KeepAspectRatio);

    int iconX = itemRect.left() + (itemRect.width() - actualSize.width()) / 2;
    int iconY = itemRect.top() + 8;
    QRect imageRect(iconX, iconY, actualSize.width(), actualSize.height());

    // Draw image (not using icon.paint, to control exact area)
    painter->drawPixmap(imageRect, pixmap);

    // === 3. Draw text label
    QString text = index.data(Qt::DisplayRole).toString();
    int textTop = imageRect.bottom() + 4;  // small margin below image
    QRect textRect(itemRect.left(), textTop, itemRect.width(), itemRect.bottom() - textTop - 2);
    painter->setPen(Qt::black);
    painter->drawText(textRect, Qt::AlignCenter, text);

    // === 4. Draw checkbox (only for files)
    QString path = index.data(Qt::ToolTipRole).toString();
    if (!path.isEmpty() && !QFileInfo(path).isDir()) {
        const int checkboxSize = 16;
        QRect checkRect(
            imageRect.right() - checkboxSize - 4,
            imageRect.top() + 4,
            checkboxSize,
            checkboxSize
            );

        bool checked = index.data(Qt::CheckStateRole).toInt() == Qt::Checked;

        QStyleOptionButton checkOpt;
        checkOpt.rect = checkRect;
        checkOpt.state = QStyle::State_Enabled | (checked ? QStyle::State_On : QStyle::State_Off);
        QApplication::style()->drawControl(QStyle::CE_CheckBox, &checkOpt, painter);
    }

    painter->restore();
}

bool FaceListItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model,
                                       const QStyleOptionViewItem& option, const QModelIndex& index) {
    QString path = index.data(Qt::ToolTipRole).toString();
    if (path.isEmpty() || QFileInfo(path).isDir()) return false;

    if (event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        QRect itemRect = option.rect;

        QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
        QSize iconSize = option.decorationSize;
        QPixmap pixmap = icon.pixmap(iconSize);
        QSize actualSize = pixmap.size().scaled(iconSize, Qt::KeepAspectRatio);

        int iconX = itemRect.left() + (itemRect.width() - actualSize.width()) / 2;
        int iconY = itemRect.top() + 8;
        QRect imageRect(iconX, iconY, actualSize.width(), actualSize.height());

        const int checkboxSize = 16;
        QRect checkRect(
            imageRect.right() - checkboxSize - 4,
            imageRect.top() + 4,
            checkboxSize,
            checkboxSize
            );

        if (checkRect.contains(mouseEvent->pos())) {
            Qt::CheckState state = static_cast<Qt::CheckState>(index.data(Qt::CheckStateRole).toInt());
            Qt::CheckState newState = (state == Qt::Checked) ? Qt::Unchecked : Qt::Checked;
            model->setData(index, newState, Qt::CheckStateRole);
            return true;
        }
    }

    return QStyledItemDelegate::editorEvent(event, model, option, index);
}
