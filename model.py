import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import math
import random
import csv
import cv2
import os
import json
import h5py

from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import callbacks



def get_csv_data(training_file):
    """
    Utility function to load training data from a csv file and
    return data as a python list.

    param: path of csv file containing the data
    """
    with open(training_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        data_list = list(reader)

    f.close()

    return data_list



def generate_batch(data_list, batch_size=64):
    images = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    OFFSETS = [0, .25, -.25]
    while 1:
        straight_count = 0
        for i in range(batch_size):
            row = random.randrange(len(data_list))
            # Limit angles of less than absolute value of .1 to no more than 1/3 of data
            # to reduce bias of car driving straight
            if abs(float(data_list[row][3])) < .1:
                straight_count += 1
            if straight_count > math.floor(batch_size * .33):
                while abs(float(data_list[row][3])) < .1:
                    row = random.randrange(len(data_list))
            image_index = random.randrange(len(OFFSETS))
            image = cv2.imread('data/' + str(data_list[row][image_index]).strip())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = process_image(image)
            image = np.array(image, dtype=np.float32)
            angle = float(data_list[row][3]) + OFFSETS[image_index]
            # Flip image 50% of the time
            if np.random.randint(2) == 0:
                image = cv2.flip(image, 1)
                angle = -angle
            images[i] = image
            angles[i] = angle
        yield images, angles


def resize(image):
    return cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)


def normalize(image):
    return image / 127.5 - 1.


def crop_image(image):
    return image[40:-20,:]


def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def process_image(image):
    image = random_brightness(image)
    image = crop_image(image)
    image = resize(image)
    return image


def get_model():

    model = Sequential([
        # Normalize image to -1.0 to 1.0
        Lambda(normalize, input_shape=(66, 200, 3)),
        # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation 
        Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .1 (keep probability of .9)
        Dropout(.1),
        # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.2),
        # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Flatten
        Flatten(),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.3),
        # Fully-connected layer 1 | 100 neurons | elu activation
        Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Fully-connected layer 2 | 50 neurons | elu activation
        Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Fully-connected layer 3 | 10 neurons | elu activation
        Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.5),
        # Output
        Dense(1, init='he_normal')
    ])
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    return model
    


if __name__=="__main__":

    # Get the Udacity provided training data from file and save it in a list
    training_file = 'data/driving_log.csv'
    # NOTE: To train using student generated dataset with over 100k examples uncomment below line
    # training_file = 'data/session_data/driving_log.csv'
    data_list = get_csv_data(training_file)
    # Shuffle the data and split into train and validation sets
    data_list = shuffle(data_list)
    training_list = data_list[:math.floor(len(data_list)*.9)]
    validation_list = data_list[math.floor(len(data_list)*.9):]

    # Stop training if the validation loss doesn't improve for 5 consecutive epochs
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    # Get model and train using fit generator due to memory constraints
    model = get_model()
    model.fit_generator(generate_batch(training_list), samples_per_epoch=24000, nb_epoch=200, validation_data=generate_batch(validation_list), nb_val_samples=1024, callbacks=[early_stop])

    print('Saving model weights and configuration file.')
    # Save model weights
    model.save_weights('model.h5')
    # Save model architecture as json file
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


    from keras import backend as K 

    K.clear_session()
