import pickle
import tensorflow as tf
import numpy as np
import cv2
import csv
import os
import json

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

    for line in data:
        image_names.append(line[0])
        steering_angles.append(line[3])
        image_names.append(line[1])
        steering_angles.append(line[3])
        image_names.append(line[2])
        steering_angles.append(line[3])

    images = []

    for name in image_names:
        image = 'data/' + str(name)
        img = cv2.imread(image)
        images.append(img)

    images = np.array(images)
    steering_angles = np.asarray(steering_angles, dtype=np.float32())

    return images, steering_angles

def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))

def normalize(image):
    return image / 127.5 - 1.


def get_model():

    img_in = Input(shape=(160, 320, 3), name='img_in')
    angle_in = Input(shape=(1,), name='angle_in')

    #x = Lambda(lambda x: x/127.5 - 1.)(img_in)
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
    ch, row, col = 3, 160, 320 # image shape

    model = Sequential()
    model.add(Lambda(lambda x: (x/127.5 - 1.), input_shape=(ch, row, col)))
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
    """

    return model


if __name__=="__main__":
    X_train, y_train = load_data()
    print(X_train.shape)
    print(X_train.shape[0])
    #X_train = X_train.reshape(X_train.shape[0], 3, 160, 320)
    #X_train = X_train.astype('float32')

    X_train_practice = X_train[:10]
    y_train_practice = y_train[:10]
    X_val_practice = X_train[10:20]
    y_val_practice = y_train[10:20]

    print(X_train.shape)
    print(X_train_practice[0].shape)

    model = get_model()
    #model.fit(X_train_practice, y_train_practice, nb_epoch=10)
    #model.fit_generator((X_train_practice, y_train_practice), samples_per_epoch=2, nb_epoch=10)




