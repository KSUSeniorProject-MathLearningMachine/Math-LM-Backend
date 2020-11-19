import numpy as np
import argparse
import cv2
from keras.models import load_model
import os
import random
import skimage.morphology

DARK_THRESH = 120
VISUALIZATION_COLOR = (0, 255, 0)
BOUNDING_BOX_PADDING = 5
INPUT_SIZE = (32, 32)
MIN_CONF = 0.50

VISUALIZE = 0

try:
    VISUALIZE = int(os.environ['VISUALIZE'])
except KeyError:
    VISUALIZE = 0


def info(msg):
    print("[INFO] {}".format(msg))


def init_model(model_dir):
    """Return the loaded model from disk"""
    return load_model(model_dir)


def init_labels(labels_file):
    return np.load(labels_file)


def classify(image, model, labels):
    # img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_GRAY2BGR)

    out = model.predict(image)

    prediction_index = out.argmax(axis=-1)[0]
    prediction = labels[prediction_index]
    confidence = out[0][prediction_index]

    info("Classification: {}, confidence: {}".format(prediction, confidence))

    return prediction, confidence


def visualization_color(index):
    random.seed(index)
    min = 100
    max = 230
    return random.randint(min, max), random.randint(min, max), random.randint(min, max)


def bounding_box(image, number):
    # Inspired by: https://stackoverflow.com/a/31402351
    image = (image == number)
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return (xmin, ymin), (xmax, ymax)


def detect(image, model, labels):
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

    image_map = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

    # convert image to scikit-image
    for xPos, y in enumerate(image):
        for yPos, (r, g, b) in enumerate(y):
            image_map[xPos][yPos] = (int(r) + int(g) + int(b)) / 3 < DARK_THRESH

    grouped_image, num_groups = skimage.morphology.label(image_map, return_num=True, connectivity=2)

    if VISUALIZE > 0:
        # Show the labelled groupings
        clone = image.copy()

        for yPos, y in enumerate(grouped_image):
            for xPos, label in enumerate(y):
                if label > 0:
                    cv2.rectangle(clone, (xPos, yPos), (xPos, yPos), visualization_color(label), 1)

        cv2.imshow("After grouping", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if VISUALIZE > 0:
        clone = image.copy()

        for group_number in range(1, num_groups + 1):
            start, end = bounding_box(grouped_image, group_number)
            cv2.rectangle(clone, tuple(i - BOUNDING_BOX_PADDING for i in start), tuple(i + BOUNDING_BOX_PADDING for i in end), VISUALIZATION_COLOR, 2)

        cv2.imshow("Bounding boxes", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    detections = []
    overall_confidence = 1.0

    for group_number in range(1, num_groups + 1):
        box = top_left, bottom_right = bounding_box(grouped_image, group_number)

        grouped_image_clone = (grouped_image == group_number)
        cropped = grouped_image_clone[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        smallest_dim = np.argmin(cropped.shape)
        sm_dim_size = cropped.shape[smallest_dim]
        lg_dim_size = np.max(cropped.shape)

        dim_diff = lg_dim_size - sm_dim_size

        if smallest_dim == 0:
            padding = np.zeros((int(dim_diff / 2.0), lg_dim_size))
        else:
            padding = np.zeros((lg_dim_size, int(dim_diff / 2.0)))

        cropped = np.concatenate((padding, cropped, padding), axis=smallest_dim)
        cropped = cv2.resize(cropped, INPUT_SIZE)

        img = (255.0 - cropped * 255.0)

        if VISUALIZE > 0:
            cv2.imshow("Cropped image", img)
            cv2.waitKey(0)

        img = np.expand_dims(img, axis=2)
        img = cv2.merge([img, img, img])

        img = np.expand_dims(img, axis=0)

        detection, confidence = classify(img, model, labels)

        if confidence > MIN_CONF:
            detections.append((box, detection, confidence))
            overall_confidence *= confidence

    VISUALIZE > 0 and cv2.destroyAllWindows()

    # TODO: test close-by groups

    return detections, overall_confidence


def init_from_cmd():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to saved model")
    ap.add_argument("-l", "--labels", required=True,
                    help="path to saved labels")
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    ap.add_argument("-v", "--visualize", type=int, default=0,
                    help="whether or not to show extra visualizations for debugging")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    if image is None:
        raise FileNotFoundError("Cannot load image: {}".format(args["image"]))

    global VISUALIZE

    try:
        VISUALIZE = int(args['visualize'] or os.environ['VISUALIZE'])
    except KeyError:
        VISUALIZE = 0

    info("Visualization level: {:d}".format(VISUALIZE))

    # TODO: Apply filters, if necessary (monochrome)

    model = init_model(args["model"])
    labels = init_labels(args["labels"])

    detections, overall_confidence = detect(image, model, labels)

    print(detections)
    print(overall_confidence)


if __name__ == '__main__':
    """If the script is run from the command line..."""
    init_from_cmd()

