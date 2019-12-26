#ifndef PIXMAP_H
#define PIXMAP_H

#include <QPixmap>
#include <QObject>
#include <QPainter>
#include <QQuickPaintedItem>

class Pixmap : public QQuickPaintedItem {
    Q_OBJECT
    Q_PROPERTY(QPixmap pixmap READ getPixmap WRITE setPixmap NOTIFY pixmapChanged)

public:
    Pixmap(QQuickItem* parent = nullptr);
    void paint(QPainter* painter) override;

    QPixmap getPixmap() const;
    void setPixmap(const QPixmap &value);

signals:
    void pixmapChanged();

private:
    QPixmap pixmap;
    QPixmap scaledPixmap;

};

#endif // PIXMAP_H
