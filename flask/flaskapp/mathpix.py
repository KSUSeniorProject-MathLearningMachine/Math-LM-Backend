import requests
import os

APP_ID = os.environ['MATHPIX_APP_ID']
APP_KEY = os.environ['MATHPIX_APP_KEY']

BASE_URL = "https://api.mathpix.com/v3"
HEADERS = {"Content-Type": "application/json", "app_id": APP_ID, "app_key": APP_KEY}


class MathpixApiException(Exception):
    pass

def submit_text(b64_image):
    data = { "src": b64_image }
    url = BASE_URL + "/text"
    response = requests.post(url, headers=HEADERS, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise MathpixApiException(response.json()['message'])
