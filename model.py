import tensorflow as tf
import numpy as np
import random, csv, cv2, json, h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.regularizers import l2


def get_csv_data(log_file):
    """
    Reads a csv file and returns two lists separated into examples and labels.
    :param log_file: The path of the log file to be read.
    """
    image_names, steering_angles = [], []
    # Steering offset used for left and right images
    steering_offset = 0.225
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            image_names.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+steering_offset, angle-steering_offset])

    return image_names, steering_angles


def generate_batch(X_train, y_train, batch_size=64):
    """
    Return two numpy arrays containing images and their associated steering angles.
    :param X_train: A list of image names to be read in from data directory.
    :param y_train: A list of steering angles associated with each image.
    :param batch_size: The size of the numpy arrays to be return on each pass.
    """
    images = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    while True:
        straight_count = 0
        for i in range(batch_size):
            # Select a random index to use for data sample
            sample_index = random.randrange(len(X_train))
            image_index = random.randrange(len(X_train[0]))
            angle = y_train[sample_index][image_index]
            # Limit angles of less than absolute value of .1 to no more than 1/2 of data
            # to reduce bias of car driving straight
            if abs(angle) < .1:
                straight_count += 1
            if straight_count > (batch_size * .5):
                while abs(y_train[sample_index][image_index]) < .1:
                    sample_index = random.randrange(len(X_train))
            # Read image in from directory, process, and convert to numpy array
            image = cv2.imread('data/' + str(X_train[sample_index][image_index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = process_image(image)
            image = np.array(image, dtype=np.float32)
            # Flip image and apply opposite angle 50% of the time
            if random.randrange(2) == 1:
                image = cv2.flip(image, 1)
                angle = -angle
            images[i] = image
            angles[i] = angle
        yield images, angles


def resize(image):
    """
    Returns an image resized to match the input size of the network.
    :param image: Image represented as a numpy array.
    """
    return cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)


def normalize(image):
    """
    Returns a normalized image with feature values from -1.0 to 1.0.
    :param image: Image represented as a numpy array.
    """
    return image / 127.5 - 1.


def crop_image(image):
    """
    Returns an image cropped 40 pixels from top and 20 pixels from bottom.
    :param image: Image represented as a numpy array.
    """
    return image[40:-20,:]


def random_brightness(image):
    """
    Returns an image with a random degree of brightness.
    :param image: Image represented as a numpy array.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def process_image(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    image = random_brightness(image)
    image = crop_image(image)
    image = resize(image)
    return image


def get_model():
    """
    Returns a compiled keras model ready for training.
    """
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
        # Dropout with drop probability of .3 (keep probability of .7)
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
        Dense(1, activation='linear', init='he_normal')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model    


if __name__=="__main__":
    # Get the training data from log file, shuffle, and split into train/validation datasets
    X_train, y_train = get_csv_data('data/driving_log.csv')
    X_train, y_train = shuffle(X_train, y_train, random_state=14)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=14)

    # Get model, print summary, and train using a generator
    model = get_model()
    model.summary()
    model.fit_generator(generate_batch(X_train, y_train), samples_per_epoch=24000, nb_epoch=30, validation_data=generate_batch(X_validation, y_validation), nb_val_samples=1024)#, callbacks=[early_stop])

    print('Saving model weights and configuration file.')
    # Save model weights
    model.save_weights('model.h5')
    # Save model architecture as json file
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


    from keras import backend as K 

    K.clear_session()
