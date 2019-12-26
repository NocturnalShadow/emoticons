import QtQuick 2.7
import QtQuick.Controls 2.14
import QtQuick.Layouts 1.12

import Qt.labs.folderlistmodel 2.12
import Qt.labs.platform 1.1

import Pixmap 1.0

ColumnLayout {
    implicitWidth: parent.implicitWidth
    height: parent.height

    property real surfaceViewportRatio: 1.5
    property real imageDefaultSize: 200
    property var imageNameFilters : ["*.png", "*.jpg", "*.gif"];
    property string picturesLocation : "";

    RowLayout {
        z: 1
        FileDialog {
            id: fileDialog
            title: "Choose an image"
            folder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0]
            onAccepted: {
                imagePath.text = file
                imageProcessorModel.updateFaces(file)
            }
        }
        TextField {
            id: imagePath
            text: ""
            Layout.fillWidth: true
        }
        Button {
            text: "Select image"
            onClicked: fileDialog.open()
        }
    }

    ListView {
        implicitWidth: parent.implicitWidth
        implicitHeight: parent.height
        Layout.fillWidth: true
        clip: true

        model: imageProcessorModel
        delegate: ColumnLayout {
            width: parent.width
//            Image {
//                source: model.image
//                fillMode: Image.PreserveAspectFit
//                Layout.fillWidth: true
//                sourceSize.width: parent.width
//            }
            Pixmap {
                width: parent.width
                height: 0
                Layout.fillWidth: true
                pixmap: model.image
            }
            Text {
                text: "(" + model.label + ")"
                color: "white"
                Layout.fillWidth: true
            }
        }
        highlight: Rectangle {
            color: 'grey'
        }
        spacing: 10
    }
}

