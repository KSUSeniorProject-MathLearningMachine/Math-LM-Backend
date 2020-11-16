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

WIDTH = 600
PYR_SCALE = 1.2
WIN_STEP = 16
ROI_SIZE = (200, 200)
INPUT_SIZE = (45, 45)
MIN_CONF = 0.98
VISUALIZE = 0

try:
    VISUALIZE = os.environ['VISUALIZE']
except KeyError:
    VISUALIZE = 0


def init_model():
    classes = 82
    LR = 1e-3
    EPOCHS = 12
    img_rows, img_cols = 45, 45
    input_shape = (img_rows, img_cols, 3)

    graph = tf.Graph()

    with graph.as_default():
        model = Sequential()

        model.add(Conv2D(32, (2, 2), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # load woeights into new model
        model.load_weights("weights.h5")
        print("Loaded Model from disk")

        # compile and evaluate loaded model
        opt = Adam(lr=LR, decay=LR / EPOCHS)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        # print('loss:', loss)
        # print('accuracy:', accuracy)

        return model, graph


def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[0], x:x + ws[0]])


def image_pyramid(image, scale=0.5, minSize=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[0] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[1] < minSize[1] or image.shape[0] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def non_max_suppression_with_labels(boxes, labels, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    boxes_with_labels = []

    for picked in pick:
        boxes_with_labels.append((boxes[picked].astype("int"), labels[picked]))


    # return only the bounding boxes that were picked
    return boxes_with_labels


def detect(image_array):
    image_array = imutils.resize(image_array, width=WIDTH)
    (H, W) = image_array.shape[:2]

    # initialize the image pyramid
    pyramid = image_pyramid(image_array, scale=PYR_SCALE, minSize=ROI_SIZE)

    # initialize two lists, one to hold the ROIs generated from the image
    # pyramid and sliding window, and another list used to store the
    # (x, y)-coordinates of where the ROI was in the original image
    rois = []
    locs = []

    # time how long it takes to loop over the image pyramid layers and
    # sliding window locations
    start = time.time()

    # loop over the image pyramid
    for image in pyramid:
        # determine the scale factor between the *original* image
        # dimensions and the *current* layer of the pyramid
        scale = W / float(image.shape[1])

        # for each layer of the image pyramid, loop over the sliding
        # window locations
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # scale the (x, y)-coordinates of the ROI with respect to the
            # *original* image dimensions
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)

            # take the ROI and pre-process it so we can later classify
            # the region using Keras/TensorFlow
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)

            # update our list of ROIs and associated coordinates
            rois.append(roi)
            locs.append((x, y, x + w, y + h))

            # check to see if we are visualizing each of the sliding
            # windows in the image pyramid
            if VISUALIZE > 1:
                # clone the original image and then draw a bounding box
                # surrounding the current region
                clone = image_array.copy()
                cv2.rectangle(clone, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

                # show the visualization and current ROI
                cv2.imshow("Visualization", clone)
                cv2.imshow("ROI", roiOrig)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

    # show how long it took to loop over the image pyramid layers and
    # sliding window locations
    end = time.time()
    print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
        end - start))

    # convert the ROIs to a NumPy array
    rois = np.array(rois, dtype="float32")

    # classify each of the proposal ROIs using ResNet and then show how
    # long the classifications took
    print("[INFO] classifying ROIs...")
    start = time.time()

    model, graph = init_model()

    preds = []
    for (roi) in rois:
        preds.append(classify(roi, model, graph))

    # preds = np.array(preds, dtype="float32")

    end = time.time()
    print("[INFO] classifying ROIs took {:.5f} seconds".format(
        end - start))

    # decode the predictions and initialize a dictionary which maps class
    # labels (keys) to any ROIs associated with that label (values)
    # preds = imagenet_utils.decode_predictions(preds, top=1)

    labels = {}

    # loop over the predictions
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p

        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if prob >= MIN_CONF:
            # grab the bounding box associated with the prediction and
            # convert the coordinates
            box = locs[i]

            # grab the list of predictions for the label and add the
            # bounding box and probability to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L

    labels_with_prob = []
    overall_confidence = 1.0

    # loop over the labels for each of detected objects in the image
    for label in labels.keys():
        # clone the original image so that we can draw on it
        print("[INFO] showing results for '{}'".format(label))
        clone = image_array.copy()

        # loop over all bounding boxes for the current label
        for (box, prob) in labels[label]:
            # draw the bounding box on the image
            (startX, startY, endX, endY) = box
            cv2.rectangle(clone, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

        # show the results *before* applying non-maxima suppression, then
        # clone the image again so we can display the results *after*
        # applying non-maxima suppression
        if VISUALIZE > 0:
            cv2.imshow("Before", clone)

        clone = image_array.copy()

        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)

        # loop over all bounding boxes that were kept after applying
        # non-maxima suppression
        for (startX, startY, endX, endY) in boxes:
            # draw the bounding box and label on the image
            cv2.rectangle(clone, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # labels_with_prob.append((label, (startX, startY), (endX, endY), proba[0]))
            # overall_confidence *= proba[0]

        # show the output after apply non-maxima suppression
        if VISUALIZE > 0:
            cv2.imshow("After", clone)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    clone = image_array.copy()

    all_boxes = []

    for label in labels:
        for (box, prob) in labels[label]:
            all_boxes.append((box, prob, label))

    # loop over all bounding boxes for the current label
    for (box, prob, label) in all_boxes:
        # draw the bounding box on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)

    # show the results *before* applying non-maxima suppression, then
    # clone the image again so we can display the results *after*
    # applying non-maxima suppression
    if VISUALIZE > 0:
        cv2.imshow("Before (all labels)", clone)

    clone = image_array.copy()

    boxes = np.array([p[0] for p in all_boxes])
    proba = np.array([p[1] for p in all_boxes])
    labels = np.array([p[2] for p in all_boxes])

    boxes = non_max_suppression_with_labels(boxes, labels, proba)

    # loop over all bounding boxes that were kept after applying
    # non-maxima suppression
    for ((startX, startY, endX, endY), label) in boxes:

        # draw the bounding box and label on the image
        cv2.rectangle(clone, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        labels_with_prob.append((label, (startX, startY), (endX, endY), proba[0]))
        overall_confidence *= proba[0]

    # show the output after apply non-maxima suppression
    if VISUALIZE > 0:
        cv2.imshow("After (all labels)", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return labels_with_prob, overall_confidence


def classify(image_array, model, graph):
    mlb = pickle.loads(open('grants.pickle', 'rb').read())

    image_array = image_array.astype("float") / 255.0
    image_array = img_to_array(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    with graph.as_default():
        out = model.predict(image_array)[0]
        prediction_index = out.argmax(axis=-1)
        prediction = ' '.join(mlb.classes_[prediction_index])
        confidence = out[prediction_index]

        if VISUALIZE > 0:
            print(prediction)

        return prediction_index, prediction, confidence


if __name__ == '__main__':
    """If the script is run from the command line..."""

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    # ap.add_argument("-s", "--size", type=str, default="(200, 150)",
    #                 help="ROI size (in pixels)")
    # ap.add_argument("-c", "--min-conf", type=float, default=0.9,
    #                 help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--visualize", type=int, default=0,
                    help="whether or not to show extra visualizations for debugging")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    if image is None:
        raise Exception('File not found')

    try:
        VISUALIZE = args['visualize'] or os.environ['VISUALIZE']
    except KeyError:
        VISUALIZE = 0

    detect(image)
