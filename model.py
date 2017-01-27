import pickle
import tensorflow as tf
import numpy as np
import cv2
import csv

from keras.layers import Input, Flatten, Dense
from keras.models import Model

def load_data():
    """
    Utility function to load training data from driving_log.csv file.
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
