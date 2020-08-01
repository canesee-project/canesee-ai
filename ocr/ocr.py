
import locale
locale.setlocale(locale.LC_ALL, 'C')
import tesserocr
from PIL import Image

def ocr (image, language):
    """"returns text from image
    args: input image
     language; eg: 'ara' fro arabic, 'en' for english ... etc
    """
    result = tesserocr.image_to_text(Image.fromarray(image), lang=language)
    return result
