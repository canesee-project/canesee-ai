# @author mhashim6 on 1/23/20


import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
from bluetooth import BluetoothConnection
from camera_feed import run_video_cam, capture
from io_types import *
from object_detection.tflite_objectDetection import detect as detect_objects
from facial_recognition.face_reco import recognize as face_reco
from ocr.ocr import ocr
import time

from utils import pretty_objects, json_of_result, log

tasks = queue.Queue(0)

current_frame = None


def fetch_new_tasks(remote: BluetoothConnection):
    while True:
        line = remote.read().decode()
        tasks.put(line)
        time.sleep(0.5)


def process_tasks(remote: BluetoothConnection):
    while True:
        task = tasks.get()
        # $type_$value
        task_segments = task.split('_')
        task_type = int(task_segments[0])

        if task_type == INPUT_MODE_CHANGE:
            change_type = int(task_segments[1])
            if change_type == MODE_CHANGE_OCR:
                log('MODE_CHANGE_OCR')
                transcript = ocr(capture(), 'ara+eng')
                log('ocr: ', transcript)
                remote.send(json_of_result(RESULT_OCR, transcript))
            elif change_type == MODE_CHANGE_SCENE:
                log('MODE_CHANGE_SCENE')
            elif change_type == MODE_CHANGE_FACE_RECOGNITION:
                log('MODE_CHANGE_FACE_RECOGNITION')
                face = face_reco(capture())
                log('face: ', face)
                remote.send(json_of_result(RESULT_FACE_RECOGNITION, face)) 
            elif change_type == MODE_CHANGE_EMOTIONS:
                log('MODE_CHANGE_EMOTIONS')
            elif change_type == MODE_CHANGE_OBJECT_DETECTION:
                log('MODE_CHANGE_OBJECT_DETECTION')
                objs = pretty_objects(detect_objects(capture()))
                log('object detection results: ', objs)
                remote.send(json_of_result(RESULT_OBJECT_DETECTION, objs))
        log("process: ", task)


def detected_new_scene(frame):
    global current_frame
    current_frame = frame
    log("a new scene")
    # TODO: ease up a little.
    # time.sleep(0.5)
    # TODO: tasks.put(what?)


def new_scenes():
    run_video_cam(detected_new_scene)


def start_app():
    remote = BluetoothConnection()
    executor = ThreadPoolExecutor(3)
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(loop.run_in_executor(executor, fetch_new_tasks, remote))
    asyncio.ensure_future(loop.run_in_executor(executor, new_scenes))
    asyncio.ensure_future(loop.run_in_executor(executor, process_tasks, remote))
    loop.run_forever()


if __name__ == '__main__':
    start_app()
