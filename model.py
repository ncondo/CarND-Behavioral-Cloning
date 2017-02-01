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
from keras.models import Sequential, Model, load_model
from keras.regularizers import l2


def get_csv_data_2(training_file):
    """
    Utility function to load training data from a csv file and
    return data as a python list.

    param: path of csv file containing the data
    """
    with open(training_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip header
        data_list = list(reader)
        image_names = []
        steering_angles = []
        OFFSET = 0.2
        for row in data_list:
            # Get recorded steering angle for center image and apply offset to get left/right angles
            center_angle = float(row[3])
            left_angle = center_angle + OFFSET
            right_angle = center_angle - OFFSET
            # Append center image name and related steering angle to lists
            image_names.append(str(row[0]).strip())
            steering_angles.append(center_angle)
            # Append left image name and related steering angle to lists
            image_names.append(str(row[1]).strip())
            steering_angles.append(left_angle)
            # Append right image name and related steering angle to lists
            image_names.append(str(row[2]).strip())
            steering_angles.append(right_angle)

    f.close()

    return image_names, steering_angles


def get_csv_data(training_file):
    with open(training_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        data_list = list(reader)

    f.close()

    return data_list


def generate_batch_2(X_train, y_train, batch_size=64):
    images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    while 1:
        shuffled = list(zip(X_train, y_train))
        random.shuffle(shuffled)
        X_train, y_train = zip(*shuffled)
        for i in range(batch_size):
            image = Image.open('data/' + X_train[i])
            image = np.array(image, dtype=np.float32)
            image = random_brightness(image)
            images[i] = image
            angles[i] = y_train[i]
        yield images, angles



def generate_batch(data_list, batch_size=64):
    images = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    OFFSETS = [0, .2, -.2]
    while 1:
        for i in range(batch_size):
            data_list = shuffle(data_list)
            row = random.randrange(len(data_list))
            image_index = random.randrange(len(OFFSETS))
            #image = Image.open('data/' + str(data_list[row][image_choice]).strip())
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
    #tf.image.resize_images(image, (66, 200))


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
        #Crop area above image and car hood
        #Lambda(crop_image, input_shape=(160, 320, 3)),
        # Resize image to 66X200X3
        #Lambda(resize),
        # Normalize image to -1.0 to 1.0
        Lambda(normalize, input_shape=(66, 200, 3)),
        # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | relu activation 
        Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | relu activation
        Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | relu activation
        Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | relu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | relu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Flatten
        Flatten(),
        # Dropout with keep probability of .2
        Dropout(.2),
        # Fully-connected layer 1 | 100 neurons
        Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with keep probability of .5
        Dropout(.5),
        # Fully-connected layer 2 | 50 neurons
        Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with keep probability of .5
        Dropout(.5),
        # Fully-connected layer 3 | 10 neurons
        Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
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

    #X_train, y_train = get_csv_data(training_file)
    data_list = get_csv_data(training_file)
    data_list = shuffle(data_list)
    training_list = data_list[:math.floor(len(data_list)*.8)]
    validation_list = data_list[math.floor(len(data_list)*.8):]

    model = get_model()
    model.fit_generator(generate_batch(training_list), samples_per_epoch=24000, nb_epoch=40, validation_data=generate_batch(validation_list), nb_val_samples=1024)

    print('Saving model weights and configuration file.')

    model.save_weights('model.h5')

    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    from keras import backend as K 

    K.clear_session()


