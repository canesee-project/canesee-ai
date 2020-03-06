import cv2
import numpy as np
import tensorflow as tf


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
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    height = 48
    width = 48
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    img = np.asarray(img)
    img=cv2.resize(img,(width,height))
    img = np.expand_dims(img, axis=-1)
    #print(img.shape)
    img= img.astype('float32')
    img /= 255.0


    input_data = np.expand_dims(img, axis=0)
    #print(input_data.shape)

    if floating_model:
      input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    #print(Arabic_labels[top_k[0]])



    return Arabic_labels[top_k[0]]




if __name__ == '__main__':

    init()
    img = cv2.imread('sad.jpg')
    emotion = detect(img)
    print(emotion)
