import pytesseract
from PIL import Image

def Ocr():
    def image_file_to_string(file):
        img = Image.open(file)
        return pytesseract.image_to_string(img).encode('ascii')
