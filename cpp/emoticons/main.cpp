#include <QQmlContext>
#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include "Pixmap.h"
#include "ImageProcessor.h"
#include "ImageProcessorModel.h"

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app { argc, argv };

    QPixmap pixmap { "images/faces1.jpg" };

    qmlRegisterType<Pixmap>("Pixmap", 1, 0, "Pixmap");

    ImageProcessorModel imageProcessorModel;

    QQmlApplicationEngine engine;

    engine.rootContext()->setContextProperty("imageProcessorModel", &imageProcessorModel);
    engine.rootContext()->setContextProperty("testPixmap", QVariant::fromValue(pixmap));

    engine.load(QUrl { "qrc:/main.qml" });

    return app.exec();
}
