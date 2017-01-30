import tensorflow as tf
import numpy as np
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

def generate_data_from_file(path, batch_size=64):
    data_list = get_csv_data(path)
    OFFSET = 0.2
    images = np.zeros((batch_size, 160, 320, 3))
    angles = np.zeros(batch_size)
    i = 0
    while True:
        for line in data_list:
            center_angle = float(line[3])
            image = 'data/' + str(line[0])
            img = cv2.imread(image)
            images[i,:,:,:] = img
            angles[i] = center_angle

            images[i+1,:,:,:] = np.fliplr(img)
            angles[i+1] = -center_angle

            left_angle = center_angle + OFFSET
            image = 'data/' + str(line[1])
            img = cv2.imread(image)
            images[i+2,:,:,:] = img
            angles[i+2] = left_angle

            right_angle = center_angle + OFFSET
            image = 'data/' + str(line[2])
            img = cv2.imread(image)
            images[i+3,:,:,:] = img
            angles[i+3] = right_angle

            i += 4
            if i == batch_size:
                i = 0
                yield images, angles



def generate_data(data_list, batch_size=64):

    image_names = []
    steering_angles = []

    OFFSET = 0.2 # use offset on right and left camera images to steer car back to center of road
    for line in data_list:
        center_angle = float(line[3])
        left_angle = center_angle + OFFSET
        right_angle = center_angle - OFFSET
        # Discard steering angles of 0 to reduce bias of driving straight
        if center_angle != 0:
            image_names.append(line[0])
            steering_angles.append(center_angle)
        image_names.append(line[1])
        steering_angles.append(left_angle)
        image_names.append(line[2])
        steering_angles.append(right_angle)

    images = np.zeros((batch_size, 160, 320, 3))
    angles = np.zeros((batch_size, 1))
    while True:
        shuffle_list = list(zip(image_names, steering_angles))
        random.shuffle(shuffle_list)
        image_names, steering_angles = zip(*shuffle_list)
        for i in range(batch_size):
            image = 'data/' + str(image_names[i])
            img = cv2.imread(image)
            images[i,:,:,:] = img
            angles[i,:] = steering_angles[i]
        yield images, angles



def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))


def normalize(image):
    return image / 127.5 - 1.


def get_model():

    model = Sequential([
        # Crop area above the horizon
        #Cropping2D(cropping=((22, 0), (0, 0)), input_shape=(160, 320, 3)),
        # Resize image to 66X200X3
        Lambda(resize, input_shape=(160, 320, 3)),
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
    
    #data_list = get_csv_data('data/driving_log.csv')

    model = get_model()
    #model.fit(X_train, y_train, nb_epoch=10, batch_size=64, validation_split=.2)
    model.fit_generator(generate_data_from_file('data/driving_log.csv'), samples_per_epoch=10240, nb_epoch=10, validation_data=generate_data_from_file('data/driving_log.csv'), nb_val_samples=1024)

    print('Saving model weights and configuration file.')

    model.save_weights('model.h5')

    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    from keras import backend as K 

    K.clear_session()


