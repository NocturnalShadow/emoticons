#include "ImageProcessorModel.h"

ImageProcessorModel::ImageProcessorModel(QObject* parent)
    : QAbstractListModel { parent }
{
}

int ImageProcessorModel::rowCount(const QModelIndex& parent) const {
    // For list models only the root node (an invalid parent) should return the list's size. For all
    // other (valid) parents, rowCount() should return 0 so that it does not become a tree model.
    if (parent.isValid())
        return 0;

    return images.size();
}

QVariant ImageProcessorModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid())
        return QVariant();

    const auto& item = images.at(index.row());

    switch (role) {
    case ImageRole:
        return QVariant{ item.image };
    case LabelRole:
        return QVariant{ item.label };
    default:
        return QVariant();
    }
}

QHash<int, QByteArray> ImageProcessorModel::roleNames() const {
    QHash<int, QByteArray> names;
    names[ImageRole] = "image";
    names[LabelRole] = "label";
    return names;
}

void ImageProcessorModel::updateFaces(QString imagePath) {
    auto extractedImages = processor.extractLabledImages(imagePath);

    beginResetModel();

    if(!images.empty()) {
        beginRemoveRows(QModelIndex(), 0, images.size() - 1);
        images.clear();
        endRemoveRows();
    }

    beginInsertRows(QModelIndex(), 0, extractedImages.size());
    images = extractedImages;
    endInsertRows();

    endResetModel();
}
