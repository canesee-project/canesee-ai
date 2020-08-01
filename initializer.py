# @author mhashim6 on 2/26/20


from object_detection.tflite_objectDetection import init as init_object_detection
from facial_recognition.face_reco import init as init_face_reco
from utils import log


def init_models():
    log('init object detection...')
    init_object_detection()
    log('init facial recognition...')
    init_face_reco()
    # TODO: other models
