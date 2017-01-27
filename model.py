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
        data = list(reader)

    images = []
    steering_angle = []
    for line in data:
        images.append(line[0])
        steering_angle.append(line[3])
        images.append(line[1])
        steering_angle.append(line[3])
        images.append(line[2])
        steering_angle.append(line[3])

