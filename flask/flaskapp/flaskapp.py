import sys
from flask import Flask, request, jsonify
import requests
from sympy.parsing.latex import parse_latex
from sympy import latex
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/mathpix-ocr', methods=['POST'])
def mathpix_ocr():
    body = request.get_json()
    img = body['b64_img']
    data = {"src":"data:image/png;base64,"+img}
    headers = {"Content-Type":"application/json","app_id":"spencerb_warpmail_net_6b5480","app_key":"59e12123f380e59b4514"}
    url = "https://api.mathpix.com/v3/text"
    response = requests.post(url, headers=headers, json=data)
    return response.json(), response.status_code

@app.route('/solve-image', methods=['POST'])
def solve_image():
    body = request.get_json()
    response = requests.post('http://0.0.0.0:5000/mathpix-ocr', json=body)
    print("Response", file=sys.stdout)
    if response.status_code != 200:
        return response.text, response.status_code
    latex_styled = response.json()['latex_styled']
    new_latex = r'{}'.format(latex_styled)
    print("new latex: {}".format(new_latex))
    expr = parse_latex(new_latex)
    try:
        data = {'solved':str(latex(expr.doit())), 'input_detected':new_latex, 'confidence':response.json()['confidence']}
        print("Doit: {}".format(data), file=sys.stdout)
        return data, 200
    except Exception as e:
        return str(e), 400

@app.route('/solve-latex', methods=['POST'])
def solve_latex():
    body = request.get_json()
    latex_styled = body['latex']
    new_latex = r'{}'.format(latex_styled)
    expr = parse_latex(new_latex)
    try:
        data = {'solved':str(latex(expr.doit()))}
        return data, 200
    except Exception as e:
        return str(e), 400

if __name__=='__main__':
    app.run(host='0.0.0.0')
