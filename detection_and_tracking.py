from objectDetection import *
from objectTracking import *
import requests
import numpy as np
import cv2

CT = CentroidTracker()
# image = "photos/images.jpg"
url = "http://192.168.1.5:8080/shot.jpg"

# define the expected input shape for the model
input_w, input_h = 416, 416
# define the probability threshold for detected objects
class_threshold = 0.6
# define the anchors
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

pastObjects = OrderedDict()
# imagesPaths = [ "photos/street.jpg"]
while True:
    print("***************************************************")

    image_shot = requests.get(url)  # read image
    img_arr = np.array(bytearray(image_shot.content), dtype=np.uint8)
    image = cv2.imdecode(img_arr , 1)


    Detects, v_labels, v_scores = detect(image, input_w, input_h, class_threshold, anchors, labels )
    rects = []

    for detect in Detects:
        box = (detect.xmin, detect.ymin, detect.xmax, detect.ymax)
        rects.append(box)
        # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # cv2.imshow('image', image)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = CT.update(rects)

    # draw both the ID of the object and the centroid of the
    # object on the output frame
    v_ids = []

    for ( i, (objectId, centroid) ) in enumerate(objects.items()):
        v_ids.append(objectId)
        if (objectId not in pastObjects.keys() ) and ( centroid not in pastObjects.values() )   :
            print("%-4d %-8s %3.3f " % (objectId, v_labels[i], rounding(v_scores[i], 1)) , " ",centroid)


    pastObjects.clear()
    for key in objects.keys():
        pastObjects[key] = objects[key]

    print("----------------------------------------------------")
    draw_boxes(image, Detects, v_labels, v_scores, v_ids )

