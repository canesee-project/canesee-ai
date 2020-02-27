# @author mhashim6 on 2/3/20


import cv2
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
from typing import Callable


def run_video_cam(on_new_scene: Callable):
    cam = cv2.VideoCapture(0)
    _, first_scene = cam.read()
    on_new_scene(first_scene)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(frame_source=cam, callback=on_new_scene)


if __name__ == '__main__':
    run_video_cam(lambda frame: print("a new scene: ", len(frame)))
