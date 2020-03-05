from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

from PIL import Image

import tensorflow as tf

def init():
    global labels
    global interpreter

    labels = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()



def detect(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = 48
    width = 48
    data = image
    gray = (np.float32(data), cv2.COLOR_RGB2GRAY)
    data = np.array(gray)
    data = data.resize((width, height))


    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_k = results.argsort()[-1:][::-1]


    for i in top_k:
        if input_details:
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))


if __name__ == '__main__':

    init()
    img = Image.open('5surprised.jpg')
    detect(img)