import tesserocr
from PIL import Image

def ocr (image, language):
    """"returns text from image
    args: input image
     language; eg: 'ara' fro arabic, 'en' for english ... etc
    """
    result = tesserocr.image_to_text(image, lang= language)
    return result

"example: "
print(ocr(Image.open('wh.PNG'), 'ara'))