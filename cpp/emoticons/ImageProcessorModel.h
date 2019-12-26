#ifndef IMAGEPROCESSORMODEL_H
#define IMAGEPROCESSORMODEL_H

#include "LabeledImage.h"
#include "ImageProcessor.h"

#include <QVector>
#include <QAbstractListModel>

class ImageProcessorModel : public QAbstractListModel
{
    Q_OBJECT

public:
    enum {
        ImageRole = Qt::UserRole,
        LabelRole
    };

    explicit ImageProcessorModel(QObject* parent = nullptr);

    // Basic functionality:
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;

    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;

    virtual QHash<int, QByteArray> roleNames() const override;

    Q_INVOKABLE void updateFaces(QString imagePath);

private:
    ImageProcessor processor;
    QVector<LabeledImage> images;
};

#endif // IMAGEPROCESSORMODEL_H
