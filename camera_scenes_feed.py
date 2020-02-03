import cv2
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
from typing import Callable


def run_video_cam(new_scene_callback: Callable):
    cam = cv2.VideoCapture(0)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(frame_source=cam, callback=new_scene_callback)


if __name__ == '__main__':
    run_video_cam(lambda frame: print("a new scene: ", len(frame)))
