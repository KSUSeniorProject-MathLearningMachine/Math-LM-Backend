import base64
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
# from ..flaskapp import CharacterSegmentation
import os
import argparse
from pickle5 import pickle
import imutils
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import argparse
import imutils
import time
import cv2
import keras.models
# from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam
import os
import random

DARK_THRESH = 100
VISUALIZATION_COLOR = (0, 255, 0)

VISUALIZE = 0

try:
    VISUALIZE = int(os.environ['VISUALIZE'])
except KeyError:
    VISUALIZE = 0


def init_model():
    """Return the loaded model from disk"""
    return


def find_nearby_pixels():
    # TODO
    return


def group_strokes():
    # TODO
    return


def classify(image, model):
    # TODO
    return


def are_pixels_nearby(pixel, other_pixel, max_distance=2 ** (1 / 2)):
    delta_x = other_pixel[0] - pixel[0]
    delta_y = other_pixel[1] - pixel[1]
    return (delta_x ** 2 + delta_y ** 2) ** (1 / 2) <= max_distance


def group_pixel(ungrouped_pixel, grouped_dark_pixels, max_distance=None):
    for group_index, grouped_pixel_group in enumerate(grouped_dark_pixels):
        for grouped_pixel in grouped_pixel_group:
            if are_pixels_nearby(ungrouped_pixel, grouped_pixel, max_distance):
                grouped_pixel_group.append(ungrouped_pixel)
                return group_index

    new_pixel_group = [ungrouped_pixel]
    grouped_dark_pixels.append(new_pixel_group)


def visualization_color(index):
    random.seed(index)
    return random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)


def detect(image, model):
    # 1. Place each dark pixel in an ungrouped list
    # 2. Create a new list with grouped pixels and add the first one to the first group,
    #    then add the second one to either the first or second group if it is touching
    #    any of the pixels in the first or second groups. Remove the pixel from the ungrouped
    #    list. Repeat until there are no elements left in the ungrouped list.
    # 3. For each group of pixels, create a new "image" with only the pixels in that group,
    #    expand the shortest dimension with light pixels so that it becomes a square, then
    # 4. Send that off to the image classifier and save the result as a detection, if above
    #    a minimum confidence.
    # 5. In the case that there are multiple detections close by, combine them into one
    #    "image" and send that off to the classifier. If that is greater than a minimum
    #    confidence, replace those detections with the new combined one.

    ungrouped_dark_pixels = []

    for xPos, y in enumerate(image):
        for yPos, (r, g, b) in enumerate(y):
            if (int(r) + int(g) + int(b)) / 3 < DARK_THRESH:
                ungrouped_dark_pixels.append((xPos, yPos))

    grouped_dark_pixels = []

    clone = image.copy()

    for ungrouped_pixel in ungrouped_dark_pixels:
        # For each ungrouped dark pixel, we see if it can be added to a group,
        # otherwise, create a new group for it.
        group_index = group_pixel(ungrouped_pixel, grouped_dark_pixels, 10)
        cv2.rectangle(clone, ungrouped_pixel[::-1], ungrouped_pixel[::-1], visualization_color(group_index), 1)

    cv2.imshow("After grouping", clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    """If the script is run from the command line..."""

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    ap.add_argument("-v", "--visualize", type=int, default=0,
                    help="whether or not to show extra visualizations for debugging")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    if image is None:
        raise Exception('File not found')

    try:
        VISUALIZE = int(args['visualize'] or os.environ['VISUALIZE'])
    except KeyError:
        VISUALIZE = 0

    # TODO: Apply filters, if necessary (monochrome)

    model = init_model()

    detect(image, model)
