from flask import Flask, request
from PIL import Image
import pytesseract
import requests
app = Flask(__name__)

@app.route('/')
def hello_world():
    img = Image.open('./ocr/math.jpg')
    print(pytesseract.image_to_string(img))
    return pytesseract.image_to_string(img).encode('ascii')

@app.route('/mathpix-ocr', methods=['POST'])
def mathpix_ocr():
    body = request.json
    img_data = body['b64_img']
    body = {"src":"data:image/png;base64,"+img_data}
    headers = {"app_id":"spencerb_warpmail_net_6b5480","app_key":"59e12123f380e59b4514"}
    r = requests.post("https://api.mathpix.com/v3/text", headers=headers, data=body)
    return r.json(), r.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0')
