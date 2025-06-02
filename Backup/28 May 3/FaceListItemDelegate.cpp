#include "FaceListItemDelegate.h"
#include <QStyle>
#include <QFileInfo>

FaceListItemDelegate::FaceListItemDelegate(QObject* parent)
    : QStyledItemDelegate(parent) {}

QSize FaceListItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const {
    Q_UNUSED(option);
    Q_UNUSED(index);
    return QSize(140, 160);  // Fixed grid size
}

void FaceListItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const {
    painter->save();

    QRect itemRect = option.rect;
    const int itemWidth = itemRect.width();
    const int itemHeight = itemRect.height();

    // Draw selection indicator
    if (option.state & QStyle::State_Selected) {
        QPen pen(QColor(100, 150, 255), 2);  // Light blue border
        painter->setPen(pen);
        painter->drawRect(itemRect.adjusted(1, 1, -1, -1));
    }

    // Calculate centered icon position
    const int iconSize = 100;
    const int iconX = itemRect.left() + (itemWidth - iconSize) / 2;
    const int iconY = itemRect.top() + 10;  // Reduced top margin
    QRect iconRect(iconX, iconY, iconSize, iconSize);

    // Draw icon centered
    QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
    icon.paint(painter, iconRect, Qt::AlignCenter);

    // Draw text label at bottom
    QString text = index.data(Qt::DisplayRole).toString();
    QRect textRect(itemRect.left(), itemRect.top() + itemHeight - 25, itemWidth, 20);
    painter->setPen(Qt::black);
    painter->drawText(textRect, Qt::AlignCenter, text);

    // Draw checkbox for files only
    QString path = index.data(Qt::ToolTipRole).toString();
    if (!path.isEmpty() && !QFileInfo(path).isDir()) {
        const int checkboxSize = 16;
        // Position checkbox inside icon's top-right corner
        QRect checkRect(iconRect.right() - checkboxSize - 4,
                        iconRect.top() + 4,
                        checkboxSize,
                        checkboxSize);

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
    if (path.isEmpty() || QFileInfo(path).isDir()) {
        return false;  // Ignore folders
    }

    if (event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        QRect itemRect = option.rect;

        // Recalculate icon position (matches paint method)
        const int iconSize = 100;
        const int iconX = itemRect.left() + (itemRect.width() - iconSize) / 2;
        const int iconY = itemRect.top() + 10;
        QRect iconRect(iconX, iconY, iconSize, iconSize);

        // Calculate checkbox position
        const int checkboxSize = 16;
        QRect checkRect(
            iconRect.right() - checkboxSize - 4,
            iconRect.top() + 4,
            checkboxSize,
            checkboxSize
            );

        // Toggle checkbox if clicked
        if (checkRect.contains(mouseEvent->pos())) {
            Qt::CheckState state = static_cast<Qt::CheckState>(index.data(Qt::CheckStateRole).toInt());
            Qt::CheckState newState = (state == Qt::Checked) ? Qt::Unchecked : Qt::Checked;
            model->setData(index, newState, Qt::CheckStateRole);
            return true;
        }
    }

    return QStyledItemDelegate::editorEvent(event, model, option, index);
}
