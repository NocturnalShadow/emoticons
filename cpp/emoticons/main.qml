import QtQuick 2.14
import QtQuick.Controls 2.14
import QtQuick.Layouts 1.12

import Pixmap 1.0

ApplicationWindow {
    visible: true
    width: 480
    height: 768
    title: "Image Processor"

    ImageProcessor {
        anchors.centerIn: parent
        width: parent.width
    }
}
