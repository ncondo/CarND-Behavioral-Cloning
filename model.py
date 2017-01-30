import tensorflow as tf
import numpy as np
from PIL import Image
import random
import cv2
import csv
import os
import json
import h5py

from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model


def get_csv_data(training_file):
    """
    Utility function to load training data from a csv file and
    return data as a python list.

    param: path of csv file containing the data
    """
    with open(training_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip header
        data_list = list(reader)
    f.close()

    return data_list


def generate_batch(data_list, batch_size=64):
    images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    OFFSETS = [0, .2, -.2]
    while 1:
        for i in range(batch_size):
            row = random.randrange(len(data_list))
            image_choice = random.randrange(len(OFFSETS))
            image = Image.open('data/' + str(data_list[row][image_choice]).strip())
            image = np.array(image, dtype=np.float32)
            images[i] = image
            angles[i] = float(data_list[row][3]) + OFFSETS[image_choice]
        yield images, angles


def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))


def normalize(image):
    return image / 127.5 - 1.

def crop_image(image):
    return image[:, :, 20:-1, :]


def get_model():

    model = Sequential([
        # Crop area above the horizon
        #Cropping2D(cropping=((22, 0), (0, 0)), input_shape=(160, 320, 3)),
        Lambda(crop_image, input_shape=(160, 320, 3)),
        # Resize image to 66X200X3
        Lambda(resize),
        # Normalize image to -1.0 to 1.0
        Lambda(normalize),
        # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | relu activation 
        Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2), init='he_normal'),
        # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | relu activation
        Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2), init='he_normal'),
        # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | relu activation
        Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2), init='he_normal'),
        # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | relu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1), init='he_normal'),
        # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | relu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1), init='he_normal'),
        # Flatten
        Flatten(),
        # Dropout with keep probability of .2
        Dropout(.2),
        # Fully-connected layer 1 | 100 neurons
        Dense(100, activation='relu', init='he_normal'),
        # Dropout with keep probability of .5
        Dropout(.5),
        # Fully-connected layer 2 | 50 neurons
        Dense(50, activation='relu', init='he_normal'),
        # Dropout with keep probability of .5
        Dropout(.5),
        # Fully-connected layer 3 | 10 neurons
        Dense(10, activation='relu', init='he_normal'),
        # Dropout with keep probability of .5
        Dropout(.5),
        # Output
        Dense(1, init='he_normal')
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    return model
    


if __name__=="__main__":

    training_file = 'data/driving_log.csv'

    data_list = get_csv_data(training_file)

    model = get_model()
    model.fit_generator(generate_batch(data_list), samples_per_epoch=10240, nb_epoch=40, validation_data=generate_batch(data_list), nb_val_samples=1024)

    print('Saving model weights and configuration file.')

    model.save_weights('model.h5')

    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    from keras import backend as K 

    K.clear_session()


