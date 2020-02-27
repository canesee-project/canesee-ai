# @author mhashim6 on 2/26/20


from object_detection.objectDetection_lite import init as init_object_detection
from utils import log


def init_models():
    log('init object detection...')
    init_object_detection()
