# import time
import tflite_runtime.interpreter as tflite
import cv2
# import re
import numpy as np


interpreter = None
labels = None


def _object_position(width, height, cx, cy):

    i, j = 0, 0
    x_unit = cx / width
    y_unit = cy / height
    if 0 < x_unit < 1/3:
        i = 0
    elif 1/3 < x_unit < 2/3:
        i = 1
    elif 2/3 < x_unit < 1:
        i = 2

    if 0 < y_unit < 1/3:
        j = 0
    elif 1/3 < y_unit < 2/3:
        j = 1
    elif 2/3 < y_unit < 1:
        j = 2

    return i, j


def _getObjects(image, results, size):

    result_size = len(results)
    detected_array = []
    position = [["بأعلى اليسار", "بالأعلى", "بأعلى اليمين"],
                ["باليسار", "بالوسط", "باليمين"],
                ["بأسفل اليسار", "بالأسفل", "بأسفل اليمين"]]

    for idx, obj in enumerate(results):

        # Prepare boundary box
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * size[0])
        xmax = int(xmax * size[0])
        ymin = int(ymin * size[1])
        ymax = int(ymax * size[1])

        centroidX = (xmax + xmin) / 2
        centroidY = (ymax + ymin) / 2
        x, y = _object_position(size[0], size[1], centroidX, centroidY)
        detected_array.append( (labels[int( obj['class_id']) ], position[y][x]) )
        # print("display :: ", detected_array[idx], " : ",str(round(obj['score'] * 100, 2)) + "%")
        # print("display :: ", detected_array[idx], "  ", obj['label'])

    return detected_array


# def _load_labels(path):
#     """Loads the labels file. Supports files with or without index numbers."""
#     with open(path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         labels = {}
#         for row_number, content in enumerate(lines):
#             pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
#             if len(pair) == 2 and pair[0].strip().isdigit():
#                 labels[int(pair[0])] = pair[1].strip()
#             else:
#                 labels[row_number] = pair[0].strip()
#     return labels


def _set_input_tensor(image):

    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def _get_output_tensor(index):

    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def _prepare_image(image, shape):

    width, height, _ = np.shape(image)
    image = cv2.resize(image, dsize=shape, interpolation=cv2.INTER_CUBIC)
    # convert to numpy array
    # scale pixel values to [0, 1]
    image = image.astype('uint8')
    # image /= 255.0
    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)
    return image, width, height


def init():

    global interpreter, labels
    # define the labels
    labels = ['شخص', 'دراجة', 'سيارة', 'دراجات نارية', 'طائرة', 'حافلة', 'قطار', 'شاحنة نقل',
              'قارب', 'إشارة ضوئية', 'صنبور الاطفاء','فاضى ', 'لافتة توقف', 'عداد موقف السيارات', 'مقعد', 'طائر', 'قط',
              'الكلب','حصان', 'خروف', 'بقرة', 'فيل', 'دب', 'الحمار الوحشي', 'زرافة',"فاضى",
              'حقيبة ظهر', 'مظلة',"فاضى","فاضى", 'حقيبة يد', 'ربطة عنق', 'حقيبة سفر', 'الطبق الطائر', 'زحلوقة', 'لوح التزلج',
              'الكرة الرياضية', 'طائرة ورقية', 'مضرب البيسبول', 'قفاز البيسبول', 'لوح تزلج', 'مزلجة', 'مضرب التنس', 'زجاجة',"فاضى",
              'كأس نبيذ', 'كوب', 'فرع', 'سكين', 'ملعقة', 'وعاء', 'موز', 'تفاحة',
              'ساندويتش', 'البرتقالي', 'بروكلي', 'جزرة', 'نقانق', 'بيتزا', 'الدونات', 'كيك',
              'كرسي', 'كنبة', 'النبات المحفوظ بوعاء', 'السرير',"فاضى", 'طاولة الطعام',"فاضى","فاضى", 'الحمام', 'التلفزيون', 'حاسوب محمول',
              'الفأر', 'التحكم عن بعد', 'لوحة المفاتيح', 'الهاتف الخلوي', 'الميكروويف', 'فرن', 'محمصة خبز', 'مكتب المدير',
              'ثلاجة',"فاضى", 'كتاب', 'ساعة حائط',  'مزهرية', 'مقص', 'دمية دب', 'مجفف شعر', 'فرشاة الأسنان']

    # labels = load_labels("labelmap.txt")

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="object_detection/detect.tflite") # python relative paths suck.
    interpreter.allocate_tensors()


def detect(image):

    global labels
    global interpreter
    threshold = 0.4
    # define the expected input shape for the model
    input_w, input_h = 300, 300

    """Returns a list of detection results, each a dictionary of object info."""
    image, image_w, image_h = _prepare_image(image, (input_w, input_h))
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)    #index=250
    interpreter.invoke()

    # Get all output details
    boxes = _get_output_tensor(0)
    classes = _get_output_tensor(1)
    scores = _get_output_tensor(2)
    count = int(_get_output_tensor(3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)

    # Perform inference
    detected_array = _getObjects(image, results, (image_w, image_h))

    return detected_array


# if __name__ == "__main__":
#
#     init()
#     image = cv2.imread("photos/animal2.png")
#     image_array = np.asarray(image)
#     t1 = time.time()
#     detected_array = detect(image_array)
#     print(detected_array)
#     t2 = time.time()
#     print("time : ", t2-t1)