import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
#from translator import translate_eg_ar


def init():
    global labels, Arabic_labels
    global interpreter

    #labels = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
    #Arabic_labels = translate_eg_ar(labels)
    Arabic_labels =['غاضب', 'مشمئز', 'خائف', 'سعيد', 'طبيعى', 'حزين', 'متفاجىء']
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()



def detect(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = 720
    width = 480
    data = image
    gray = (np.float32(data), cv2.COLOR_RGB2GRAY)
    data = np.array(gray)
    data = data.resize((width, height))


    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_p = results.argsort()[-1:][::-1]

    for i in top_p :
       expression = (Arabic_labels[i])


    return expression




if __name__ == '__main__':

    init()
    img = Image.open('surprised.jpg')
    emotion = detect(img)
    print(emotion)