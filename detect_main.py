import cv2
import numpy as np
from ultralytics import YOLO

cam = cv2.VideoCapture(0)
model = YOLO("content/jetson_orinnano.engine")
print("YOLO model loaded")


def filter_results(results):
    filter = results[0]
    boxes = filter.boxes
    conf_thresh = boxes.conf > 0.75
    filter.boxes = boxes[conf_thresh]
    return filter


def get_frame_raw():
    if cam is None or not cam.isOpened():
        return None

    success, frame = cam.read()
    if not success:
        return None
    return frame


def get_frame():
    if cam is None or not cam.isOpened():
        return None

    if model is None:
        return None

    success, frame = cam.read()
    if not success:
        return None

    results = model(frame)
    filtered = filter_results(results)
    annotated_frame = filtered.plot()
    return annotated_frame


def release_camera():
    if cam is not None:
        cam.release()
        print("Camera released")
