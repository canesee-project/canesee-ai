from PIL import Image
import pytesseract as tess

#tesseract excutable path
tess.pytesseract.tesseract_cmd =r'<tesseract excuatable path>'

#image to text function
def ocr_fun(image,lang):
    img = image
    text = tess.image_to_string(img, lang=lang)
    return text

#printing the o/p text:languages are (ara, en , fra ...etc)
print(ocr_fun('<input image>','<language>'))
