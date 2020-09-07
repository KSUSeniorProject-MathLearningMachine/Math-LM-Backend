from flask import Flask, request, jsonify
import requests
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

if __name__=='__main__':
    app.run(host='0.0.0.0')
