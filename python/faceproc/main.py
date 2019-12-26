import base64
import flask

from io import BytesIO
from PIL import Image
from flask import Flask, flash, request
from mtcnn.mtcnn import MTCNN
from matplotlib import cm

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

IMG_SIZE = 48
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

app = Flask(__name__)

# create the detector, using default weights
detector = MTCNN()

# load trained facial emotions recognition model
model = keras.models.load_model('models/facial-emotions.h5')
model.summary()


def format_image(img):
    img = tf.cast(img, tf.float32)
    img = img / 127.5
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def predict_emotion(img, model):
    img = format_image(img)
    img = np.expand_dims(img, axis=0)
    images = np.vstack([img])
    classes = model.predict(images)

    return emotions[np.argmax(classes[0])]


def draw_annotated_faces(image, model):
    faces = find_faces(image)
    # plot each face as a subplot
    for i, face in enumerate(faces):
        print(str(face.shape))
        # define subplot
        plt.subplot(1, len(faces), i + 1)
        plt.gca().set_title(predict_emotion(face, model))
        plt.axis('off')
        # plot face
        plt.imshow(face)

    # show the plot
    plt.show()


def get_annotated_faces(image, model):
    faces = find_faces(image)
    result = []
    for face in faces:
        buffer = BytesIO()
        im = Image.fromarray(face)
        im.save(buffer, format="JPEG")
        result.append({
            "label": predict_emotion(face, model),
            "image": base64.b64encode(buffer.getvalue()).decode('utf-8')
        })

    return result


def crop_face(img, face_meta):
    x1, y1, width, height = face_meta['box']
    x2, y2 = x1 + width, y1 + height
    return img[y1:y2, x1:x2]


def find_faces(img):
    faces = detector.detect_faces(img)
    return [crop_face(img, faces[i]) for i in range(len(faces))]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/face', methods=['POST'])
def face():
    if 'img' not in request.files:
        flash('No image found.')
        return "No image found."

    img = request.files['img']

    if img.filename == '':
        flash('No image found.')
        return "No image found."
    if not allowed_file(img.filename):
        flash('Image extension is not recognised.')
        return "Image extension is not recognised."
    if img:
        # save bytes in a buffer
        image_bytes = BytesIO(img.read())

        # convert to Pillow Image
        img = Image.open(image_bytes)
        if not img:
            flash('Can not parse image provided.')
            return "Can not parse image provided."

        annotated_faces = get_annotated_faces(np.array(img), model)
        print(flask.jsonify(annotated_faces))

        return flask.jsonify(annotated_faces)


app.secret_key = "super secret key"

if __name__ == '__main__':
    app.run()
