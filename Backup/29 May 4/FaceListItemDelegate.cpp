#include "FaceListItemDelegate.h"
#include <QApplication>
#include <QStyle>
#include <QStyleOptionButton>

FaceListItemDelegate::FaceListItemDelegate(ViewMode m, QObject* parent)
    : QStyledItemDelegate(parent), mode(m) {}

QSize FaceListItemDelegate::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const {
    Q_UNUSED(option);
    Q_UNUSED(index);
    return QSize(140, 160);  // Match your grid size
}

void FaceListItemDelegate::paint(QPainter* painter, const QStyleOptionViewItem& option,
                                 const QModelIndex& index) const {
    painter->save();
    QRect itemRect = option.rect;

    // === 1. Selection highlight
    if (option.state & QStyle::State_Selected) {
        painter->setPen(QPen(QColor(100, 150, 255), 2));
        painter->drawRect(itemRect.adjusted(1, 1, -2, -2));
    }

    // === 2. Get icon and draw it
    QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
    QSize iconSize = option.decorationSize;
    QPixmap pixmap = icon.pixmap(iconSize);
    QSize actualSize = pixmap.size().scaled(iconSize, Qt::KeepAspectRatio);

    int iconX = itemRect.left() + (itemRect.width() - actualSize.width()) / 2;
    int iconY = itemRect.top() + (itemRect.height() - actualSize.height() - 22) / 2;
    QRect imageRect(iconX, iconY, actualSize.width(), actualSize.height());
    painter->drawPixmap(imageRect, pixmap);

    // === 3. Draw text label (e.g. "Person 1")
    QString text = index.data(Qt::DisplayRole).toString();
    QRect textRect(itemRect.left(), itemRect.bottom() - 22, itemRect.width(), 20);
    painter->drawText(textRect, Qt::AlignCenter, text);

    // === 4. Draw checkbox if needed
    bool shouldDrawCheckbox = false;
    Qt::CheckState checkState = Qt::Unchecked;

    if (mode == FaceListMode) {
        shouldDrawCheckbox = true;
    } else if (mode == FolderViewMode) {
        shouldDrawCheckbox = index.flags() & Qt::ItemIsUserCheckable;
    }

    if (shouldDrawCheckbox) {
        checkState = static_cast<Qt::CheckState>(index.data(Qt::CheckStateRole).toInt());

        QStyleOptionButton checkboxOption;
        checkboxOption.state = QStyle::State_Enabled |
                               (checkState == Qt::Checked ? QStyle::State_On : QStyle::State_Off);

        // ✅ Draw checkbox inside the imageRect (not full itemRect)
        const int checkboxSize = 16;
        QRect checkRect(
            imageRect.right() - checkboxSize - 4,
            imageRect.top() + 4,
            checkboxSize,
            checkboxSize
            );

        checkboxOption.rect = checkRect;
        QApplication::style()->drawControl(QStyle::CE_CheckBox, &checkboxOption, painter);
    }


    painter->restore();
}

bool FaceListItemDelegate::editorEvent(QEvent* event, QAbstractItemModel* model,
                                       const QStyleOptionViewItem& option, const QModelIndex& index) {
    if (!(index.flags() & Qt::ItemIsUserCheckable))
        return false;

    if (event->type() == QEvent::MouseButtonRelease) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        QRect itemRect = option.rect;

        // === Recompute imageRect same as in paint()
        QIcon icon = qvariant_cast<QIcon>(index.data(Qt::DecorationRole));
        QSize iconSize = option.decorationSize;
        QPixmap pixmap = icon.pixmap(iconSize);
        QSize actualSize = pixmap.size().scaled(iconSize, Qt::KeepAspectRatio);

        int iconX = itemRect.left() + (itemRect.width() - actualSize.width()) / 2;
        int iconY = itemRect.top() + 8;
        QRect imageRect(iconX, iconY, actualSize.width(), actualSize.height());

        // === Define checkboxRect based on imageRect
        const int checkboxSize = 16;
        QRect checkboxRect(
            imageRect.right() - checkboxSize - 4,
            imageRect.top() + 4,
            checkboxSize,
            checkboxSize
            );

        if (checkboxRect.contains(mouseEvent->pos())) {
            Qt::CheckState current = static_cast<Qt::CheckState>(index.data(Qt::CheckStateRole).toInt());
            model->setData(index, current == Qt::Checked ? Qt::Unchecked : Qt::Checked, Qt::CheckStateRole);
            return true;
        }
    }


    return false;
}
