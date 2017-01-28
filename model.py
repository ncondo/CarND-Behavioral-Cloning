import tensorflow as tf
import numpy as np
import cv2
import csv
import os
import json
import h5py

from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, Model, load_model


def load_data():
    """
    Utility function to load training data from driving_log.csv file and
    return two numpy arrays containing images and related steering angles.
    """
    training_file = 'data/driving_log.csv'

    with open(training_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip header
        data = list(reader)

    image_names = []
    steering_angles = []
    OFFSET = 0.2 # use offset on right and left camera images to steer car back to center of road
    for line in data:
        center_angle = float(line[3])
        left_angle = center_angle + OFFSET
        right_angle = center_angle - OFFSET
        image_names.append(line[0])
        steering_angles.append(center_angle)
        image_names.append(line[1])
        steering_angles.append(left_angle)
        image_names.append(line[2])
        steering_angles.append(right_angle)

    images = np.zeros((len(image_names), 160, 320, 3))

    for i in range(len(image_names)):
        image = 'data/' + str(image_names[i])
        img = cv2.imread(image)
        images[i,:,:,:] = img

    return images, np.asarray(steering_angles, dtype=np.float32())


def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))


def normalize(image):
    return image / 127.5 - 1.


def get_model():
    """
    img_in = Input(shape=(160, 320, 3), name='img_in')
    angle_in = Input(shape=(1,), name='angle_in')

    x = Lambda(resize)(img_in)
    x = Lambda(normalize)(x)
    x = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same')(x)
    x = Flatten()(x)
    x = Dropout(.2)(x)
    x = Activation('relu')(x)
    x = Dense(512)(x)
    x = Dropout(.5)(x)
    x = Activation('relu')(x)
    angle_out = Dense(1, name='angle_out')(x)

    model = Model(input=[img_in], output=[angle_out])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    """
    row, col, ch = 160, 320, 3 # image shape

    model = Sequential()
    model.add(Lambda(resize, input_shape=(row, col, ch)))
    model.add(Lambda(normalize))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


if __name__=="__main__":
    
    X_train, y_train = load_data()

    model = get_model()
    model.fit(X_train, y_train, nb_epoch=10, batch_size=64, validation_split=.2)
    #model.fit_generator((X_train_practice, y_train_practice), samples_per_epoch=2, nb_epoch=10)

    print('Saving model weights and configuration file.')

    model.save_weights('model.h5')

    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


