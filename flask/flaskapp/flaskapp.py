import latex_solver
import detector
import object_parser
from flask import Flask, request
import cv2
import numpy as np
import base64
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#MODEL = detector.init_model(os.environ['MODEL'])
LABELS = []#detector.init_labels(os.environ['LABELS'])


@app.route('/')
@cross_origin()
def hello_world():
    return 'Hello, World!'


@app.route('/ocr', methods=['POST'])
@cross_origin()
def ocr():
    im_b64 = request.get_json()['b64_img']

    im_b64 = im_b64.replace('data:image/png;base64,', '')

    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    detections, confidence, boxes = detector.detect(img, os.environ['MODEL'], LABELS)

    latex = object_parser.parse(detections)

    return {
        # "confidence": confidence,
        "latex_styled": latex,
        "detections": boxes,
        "confidence": confidence
    }

    # detections_json = [{
    #     "box": {
    #         "startX": str(startX),
    #         "startY": str(startY),
    #         "endX": str(endX),
    #         "endY": str(endY),
    #     },
    #     "label": str(detection),
    #     "confidence": str(confidence)
    # } for ((startX, startY), (endX, endY)), detection, confidence in detections]

    # return {
    #     "confidence": confidence,
    #     "latex_styled": latex,
    #     "detections": detections_json,
    # }


@app.route('/solve-image', methods=['POST'])
@cross_origin()
def solve_image():
    im_b64 = request.get_json()['b64_img']

    im_b64 = im_b64.replace('data:image/png;base64,', '')

    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    detections, overall_confidence, boxes = detector.detect(img, os.environ['MODEL'], LABELS)
    latex = object_parser.parse(detections)

    # detections_json = [{
    #     "box": {
    #         "startX": str(startX),
    #         "startY": str(startY),
    #         "endX": str(endX),
    #         "endY": str(endY),
    #     },
    #     "label": str(detection),
    #     "confidence": str(confidence)
    # } for ((startX, startY), (endX, endY)), detection, confidence in detections]

    # data = {
    #     'solved': latex_solver.solve(latex),
    #     'input_detected': latex_solver.format_latex(latex),
    #     'confidence': overall_confidence,
    #     'detections': detections_json,
    # }
    data = {
        'detections': boxes,
        'solved': latex_solver.solve(latex),
        'input_detected': latex_solver.format_latex(latex),
        'confidence': overall_confidence,
    }

    return data, 200


@app.route('/solve-latex', methods=['POST'])
@cross_origin()
def solve_latex():
    latex_styled = request.get_json()['latex']
    try:
        data = {
            'solved': latex_solver.solve(latex_styled)
        }
        return data, 200
    except Exception as e:
        return str(e), 400


if __name__=='__main__':
    app.run(host='0.0.0.0')
