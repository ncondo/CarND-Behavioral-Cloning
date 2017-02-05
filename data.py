import cv2
import numpy as np
import random


def augment_data():
    left_image = cv2.imread('example_assets/left_image_example.jpeg')
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    left_image = process_image(left_image)
    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('example_assets/left_image_example_augmented.jpeg', left_image)

    center_image = cv2.imread('example_assets/center_image_example.jpeg')
    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
    center_image = process_image(center_image)
    center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('example_assets/center_image_example_augmented.jpeg', center_image)

    right_image = cv2.imread('example_assets/right_image_example.jpeg')
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    right_image = process_image(right_image)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('example_assets/right_image_example_augmented.jpeg', right_image)


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


def crop_image(image):
    """
    Returns an image cropped 40 pixels from top and 20 pixels from bottom.
    :param image: Image represented as a numpy array.
    """
    return image[40:-20,:]


def resize(image):
    """
    Returns an image resized to match the input size of the network.
    :param image: Image represented as a numpy array.
    """
    return cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)


def process_image(image):
    """
    Returns an image after applying several preprocessing functions.
    :param image: Image represented as a numpy array.
    """
    image = random_brightness(image)
    image = crop_image(image)
    image = resize(image)
    return image


if __name__=="__main__":

    augment_data()