import numpy as np
import tensorflow as tf
import os
import time

# np.random.seed(42)

def model_details(model_path):
    interpreter=tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print ("model path: "+ model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    return invoke(interpreter,input_details,output_details)


def invoke(interpreter,input_details,output_details):
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random(input_shape), dtype = input_details[0]['dtype'])
    tic = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    toc = time.time()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print (toc-tic)
    return output_data 


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

model_details("mobilenet_1_4_224_V1_1_not_quant.tflite")
model_details("mobilenet_1_4_224_V1_1_latency.tflite")
model_details("Resnet_no_quant.tflite")
model_details("Resnet_latency.tflite")

# model_details(".tflite")
# model_details(".tflite")
# model_details(".tflite")



# Load TFLite model and allocate tensors.
# model_path="inceptionV3_quant_size.tflite"
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Test model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype = input_details[0]['dtype'])
# interpreter.set_tensor(input_details[0]['index'], input_data)
# for _ in range(5):
#     tic = time.time()
#     interpreter.invoke()
#     print ("time of {0} : {1}".format( model_path , time.time()-tic ))
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

