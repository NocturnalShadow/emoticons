#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "LabeledImage.h"

#include <QUrl>
#include <QFile>
#include <QFileInfo>
#include <QDebug>
#include <QObject>
#include <QString>
#include <QPixmap>
#include <QEventLoop>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QHttpMultiPart>
#include <QNetworkReply>
#include <QNetworkAccessManager>

#ifdef Q_OS_ANDROID
#include <QtAndroid>
#include <QtAndroidExtras/QAndroidJniObject>
#include <jni.h>
#endif

class ImageProcessor : public QObject {
   Q_OBJECT
public:
    explicit ImageProcessor (QObject* parent = nullptr)
        : QObject(parent) { }

    Q_INVOKABLE QVector<LabeledImage> extractLabledImages(QString filePath);
};


#endif // IMAGE_PROCESSOR_H
