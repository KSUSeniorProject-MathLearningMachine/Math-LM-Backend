import requests


BASE_URL = "https://api.mathpix.com/v3"
APP_ID = "spencerb_warpmail_net_6b5480"
APP_KEY = "59e12123f380e59b4514"
HEADERS = {"Content-Type": "application/json", "app_id": APP_ID, "app_key": APP_KEY}


class MathpixApiException(Exception):
    pass

def submit_text(b64_image):
    data = { "src": "data:image/png;base64," + b64_image }
    url = BASE_URL + "/text"
    response = requests.post(url, headers=HEADERS, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise MathpixApiException(response.json()['message'])
