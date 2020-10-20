import base64
from PIL import Image
from io import BytesIO
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import CharacterSegmentation as cs
import os


def detect(b64_image):
    """Apply the object detections and return a tuple of the detections as well as the confidence"""

    # load image from base64
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(b64_image)) # may not work with uri prefix
    pil_img = Image.open(sbuf)

    # segment images
    cs.image_segmentation(pil_img)

    # load segmented images. TODO: make this less janky by keeping images in memory instead of filesystem
    segmented_images = []
    files = [f for r, d, f in os.walk('/segmented/')][0]
    files = sorted(files)
    for f in files:
        segmented_images.append(Image.open('/segmented/' + f))

    model = load_model('/app/classification.model')

    predictedLabels = []
    for img in segmented_images:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (32,32))
        img = img_to_array(img)
        preds = model.predict(np.array([img]))
        labelNames = sorted(["-", "!", "(", ")", ",", "[", "]", "{", "}", "+", "=", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "alpha", "ascii_124", "b", "beta", "C", "cos", "d", "Delta", "div", "e", "exists", "f", "forall", "forward_slash", "G", "gamma", "geq", "gt", "H", "i", "in", "infty", "int", "j", "k", "l", "lambda", "ldots", "leq", "lim", "log", "lt", "M", "mu", "N", "neq", "o", "p", "phi", "pi", "pm", "prime", "q", "R", "rightarrow", "S", "sigma", "sin", "sqrt", "sum", "T", "tan", "theta", "times", "u", "v", "w", "X", "y", "z"])
        
        predictedLabel = labelNames[preds.argmax(axis=1)[0]]
        predictedLabels.append(predictedLabel)
    
    # calculate confidence
    overall_confidence = 0
    for confidence in predictedLabels:
        overall_confidence + confidence
    overall_confidence = overall_confidence / len(predictedLabels)

    return (predictedLabels, overall_confidence)
