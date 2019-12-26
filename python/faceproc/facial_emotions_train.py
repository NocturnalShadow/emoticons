# from __future__ import absolute_import, division, print_function, unicode_literals

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

IMG_SIZE = 48
IMG_SHAPE = (48, 48)
NUM_CLASSES = 7
BATCH_SIZE = 128
TRAIN_SAMPLES = 28709
VALIDATION_SAMPLES = 3589
TEST_SAMPLES = 3589
TRAIN_SAMPLES_DIR = "fer2013/Training"          # 28709
VALIDATION_SAMPLES_DIR = "fer2013/PrivateTest"  # 3589
TEST_SAMPLES_DIR = "fer2013/PublicTest"         # 3589

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutra"]
detector = MTCNN()


class TerminationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.7:
            print("\nReached 70% accuracy so cancelling training!")
            self.model.stop_training = True


def pandas_vector_to_list(pandas_df):
    py_list = [item[0] for item in pandas_df.values.tolist()]
    return py_list


def save_images(raw_data, img_size=IMG_SIZE):
    pixels_data = pandas_vector_to_list(raw_data[['pixels']])
    emotion_data = pandas_vector_to_list(raw_data[['emotion']])
    usage = pandas_vector_to_list(raw_data[['Usage']])

    for index, item in enumerate(pixels_data):
        # 48x48
        image = np.zeros((img_size, img_size), dtype=np.uint8)
        # split space separated ints
        pixel_data = item.split()
        # 0 -> 47, loop through the rows
        for i in range(0, img_size):
            # (0 = 0), (1 = 47), (2 = 94), ...
            pixel_index = i * img_size
            # (0 = [0:47]), (1 = [47: 94]), (2 = [94, 141]), ...
            image[i] = pixel_data[pixel_index:pixel_index + img_size]

        cv2.imwrite('fer2013/' + usage[index] + '/' + emotions[emotion_data[index]] + '/' + str(index) + '.png', image)


def visualize_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def format_image(img):
    img = tf.cast(img, tf.float32)
    img = img/127.5
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
    # raw_data_csv_file_name = "fer2013/fer2013.csv"
    # raw_data = pd.read_csv(raw_data_csv_file_name)
    # print(raw_data.info())
    # print(raw_data.head())
    # print(raw_data["Usage"].value_counts())
    #
    # save_images(raw_data)

    # All images will be rescaled by 1./127.5
    train_datagen = image.ImageDataGenerator(rescale=1 / 127.5)
    validation_datagen = image.ImageDataGenerator(rescale=1 / 127.5)

    # Flow training images in batches of 128 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_SAMPLES_DIR,
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_SAMPLES_DIR,
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # # Create the base model from the pre-trained model MobileNet V2
    # base_model = keras.applications.MobileNetV2(
    #     input_shape=(IMG_SIZE, IMG_SIZE, 3),
    #     include_top=False,
    #     weights='imagenet')
    #
    # base_model.trainable = False
    #
    # head_model = tf.keras.Sequential([
    #     keras.layers.GlobalAveragePooling2D(),  # input (4, 4, 1280)
    #     tf.keras.layers.Dense(256, activation='relu'),
    #     tf.keras.layers.Dense(7, activation='softmax')
    # ])

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.summary()

    base_learning_rate = 0.01
    model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=base_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint_path = "checkpoints/facial_emotions_3/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=1)

    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    # model.save('models/facial-emotions.h5')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=np.int32(TRAIN_SAMPLES / BATCH_SIZE),
        epochs=20,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=np.int32(VALIDATION_SAMPLES / BATCH_SIZE),
        callbacks=[cp_callback, TerminationCallback()])
    visualize_history(history)

    # filename = 'inputs/faces1.jpg'
    # img = plt.imread(filename)
    #
    # # create the detector, using default weights
    # # detect faces in the image
    # faces = detector.detect_faces(img)
    # # display faces on the original image
    # draw_faces(img, faces, model)
    # draw_image_with_boxes(img, faces, model)
