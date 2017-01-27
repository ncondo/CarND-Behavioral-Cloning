import pickle
import tensorflow as tf
import numpy as np
import cv2
import csv
import os
import json

from keras.layers import Dense, Dropout, ELU, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

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

    images = np.asarray(images)
    steering_angles = np.asarray(steering_angles, dtype=np.float32())

    return images, steering_angles


def get_model():
    ch, row, col = 3, 160, 320 # image shape

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(ch, row, col),
              output_shape=(ch, row, col)))
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



