import numpy as np
import argparse
import cv2
from keras.models import load_model
import os
import random
import skimage.morphology
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import functools 


def detect(image, model, labels):
    model = load_model(model)
    image = cv2.imread("image.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")
    chars = []
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    preds = model.predict(chars)

    labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-"

    labelNames = [l for l in labelNames]
    
    predictedLabels = []
    probabilities = []
    for (pred, (x,y,w,h)) in zip(preds, boxes):
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        predictedLabels.append(label)
        probabilities.append(prob)

    overall_probability = functools.reduce(operator.mul, probabilities)
    return (predictedLabels, overall_probability)


    