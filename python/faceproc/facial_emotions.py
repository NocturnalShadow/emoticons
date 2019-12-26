import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

IMG_SIZE = 48

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
detector = MTCNN()


def format_image(img):
    img = tf.cast(img, tf.float32)
    img = img / 127.5
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def predict_emotion(img, model):
    # img = image.load_img(filename, target_size=(48, 48))
    # img = image.img_to_array(img)

    img = format_image(img)
    img = np.expand_dims(img, axis=0)
    images = np.vstack([img])
    classes = model.predict(images)

    return emotions[np.argmax(classes[0])]


# draw an image with detected objects
def draw_image_with_boxes(image, result_list, model):
    # plot the image
    plt.imshow(image)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    # show the plot
    plt.show()


def draw_faces(image, result_list, model):
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        plt.subplot(1, len(result_list), i + 1)
        plt.gca().set_title(predict_emotion(image[y1:y2, x1:x2], model))
        plt.axis('off')
        # plot face
        plt.imshow(image[y1:y2, x1:x2])
    # show the plot
    plt.show()


if __name__ == '__main__':
    # model = keras.models.load_model('models/facial-emotions.h5')
    # model.summary()

    filename = 'uploads/faces1.jpg'
    img = plt.imread(filename)
    print(type(img).__name__)
    print(str(img.shape))

    # # detect faces in the image
    # faces = detector.detect_faces(img)
    # # display faces on the original image
    # draw_faces(img, faces, model)
    # draw_image_with_boxes(img, faces, model)
