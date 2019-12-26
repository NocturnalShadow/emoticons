#include "Pixmap.h"

#include <QFileInfo>
#include <QSize>
#include <QSizeF>

Pixmap::Pixmap(QQuickItem* parent)
    : QQuickPaintedItem { parent }
{
}

void Pixmap::paint(QPainter* painter) {
    if(pixmap.isNull())
        return;

    qDebug() << Q_FUNC_INFO << "invoked...";
    qDebug() << "Item rect size:" << boundingRect();

//    auto center = boundingRect().center() - scaledPixmap.rect().center();

//    if (center.x() < 0) center.setX(0);
//    if (center.y() < 0) center.setY(0);

    painter->drawPixmap(0, 0, scaledPixmap);
}

QPixmap Pixmap::getPixmap() const {
    return pixmap;
}

void Pixmap::setPixmap(const QPixmap& value) {
    if (&pixmap == &value)
        return;

    pixmap = value;
    scaledPixmap = value.scaledToWidth(static_cast<int>(boundingRect().width()));

    this->setHeight(scaledPixmap.height());

    qDebug() << "Image size:" << pixmap.size();
    qDebug() << "Scaled Image size:" << scaledPixmap.size();
    qDebug() << "Item size:" << size();

    update();
}
