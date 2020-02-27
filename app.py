import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
from bluetooth import BluetoothConnection
from camera_scenes_feed import run_video_cam
from io_types import *
from object_detection.objectDetection_lite import detect as detect_objects
import time

tasks = queue.Queue(0)

current_frame = None


def fetch_new_tasks(remote: BluetoothConnection):
    while True:
        line = remote.read().decode()
        tasks.put(line)


def process_tasks(remote: BluetoothConnection):
    while True:
        task = tasks.get()
        # $type_$value
        task_segments = task.split('_')
        task_type = int(task_segments[0])
        if task_type == INPUT_MODE_CHANGE:
            change_type = int(task_segments[1])
            if change_type == MODE_CHANGE_OCR:
                print('MODE_CHANGE_OCR')
            elif change_type == MODE_CHANGE_SCENE:
                print('MODE_CHANGE_SCENE')
            elif change_type == MODE_CHANGE_FACE_RECOGNITION:
                print('MODE_CHANGE_FACE_RECOGNITION')
            elif change_type == MODE_CHANGE_EMOTIONS:
                print('MODE_CHANGE_EMOTIONS')
            elif change_type == MODE_CHANGE_OBJECT_DETECTION:
                print('MODE_CHANGE_OBJECT_DETECTION')
        print("process: ", task)


def detected_new_scene(frame):
    global current_frame
    current_frame = frame
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
