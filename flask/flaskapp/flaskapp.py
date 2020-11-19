import latex_solver
import detector
import object_parser
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ocr', methods=['POST'])
def ocr():
    img = request.get_json()['b64_img']

    detections, confidence = detector.detect(img)

    latex = object_parser.parse(detections)

    return {
        "confidence": confidence,
        "latex_styled": latex,
    }

@app.route('/solve-image', methods=['POST'])
def solve_image():
    img = request.get_json()['b64_img']

    detections, confidence = detector.detect(img)
    latex = object_parser.parse(detections)

    data = {
        'solved': latex_solver.solve(latex),
        'input_detected': latex_solver.format_latex(latex),
        'confidence': confidence
    }

    return data, 200

@app.route('/solve-latex', methods=['POST'])
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
