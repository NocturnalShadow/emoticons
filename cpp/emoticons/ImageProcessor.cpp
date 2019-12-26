#include "ImageProcessor.h"

#include <QByteArray>

inline LabeledImage deserializeLabeledImage(QJsonValueRef json) {
    auto jsonObj = json.toObject();
    auto imageBase64 = jsonObj["image"].toString().toUtf8();

    QPixmap image;
    image.loadFromData(QByteArray::fromBase64(imageBase64), "JPG");

    return { image, jsonObj["label"].toString() };
}

QVector<LabeledImage> ImageProcessor::extractLabledImages(QString filePath) {
    QVector<LabeledImage> result;

#ifdef Q_OS_ANDROID
    QtAndroid::PermissionResultMap permisionsResult = QtAndroid::requestPermissionsSync(QStringList({"android.permission.READ_EXTERNAL_STORAGE"}));
    if (permisionsResult["android.permission.READ_EXTERNAL_STORAGE"] == QtAndroid::PermissionResult::Denied) {
        qDebug() << "ERROR: permissions denied.";
        return result;
    }

    QAndroidJniObject jsPath = QAndroidJniObject::fromString(filePath);
    QAndroidJniObject path = QAndroidJniObject::callStaticObjectMethod(
                "org.qtproject.utils.QPathResolver",
                "getRealPathFromURI",
                "(Landroid/content/Context;Ljava/lang/String;)Ljava/lang/String;",
                QtAndroid::androidActivity().object(), jsPath.object());

    filePath = path.toString();
#elif defined Q_OS_WINDOWS
    QUrl fileUrl{ filePath };
    filePath = fileUrl.toLocalFile();
#endif

    auto networkManager = new QNetworkAccessManager();
    auto multiPart = new QHttpMultiPart { QHttpMultiPart::FormDataType };
    auto file = new QFile { filePath };
    file->open(QIODevice::ReadOnly);
    file->setParent(multiPart);

    QFileInfo fileInfo { filePath };
    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant(QString("form-data; name=\"img\";filename=\"%1\"").arg(fileInfo.fileName()).toLatin1()) );
    imagePart.setBodyDevice(file);

    multiPart->append(imagePart);

    QUrl imgProcessorUrl { "http://localhost:5000/face" };
    QNetworkRequest request { imgProcessorUrl };
    auto reply = networkManager->post(request, multiPart);

    QEventLoop loop;
    connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    if(reply->error() == QNetworkReply::NoError) {
        auto response = (QString) reply->readAll();
        auto jsonResponse = QJsonDocument::fromJson(response.toUtf8());
        auto jsonArray = jsonResponse.array();
        for (auto jsonValue : jsonArray) {
            result << deserializeLabeledImage(jsonValue);
        }
    } else {
        qDebug() << "ERROR: " << reply->errorString();
    }

    return result;
}
