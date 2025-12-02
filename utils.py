import json
from math import atan

import cv2
import numpy


def calculate_yaw(bbox, f_x, img_width=640):
    x1, y1, x2, y2 = bbox.xyxy[0]
    x = (x1 + x2) / 2
    x -= img_width
    return atan(x / f_x)


def get_calibration_camera_matrix(path):
    with open(path) as f:
        data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    return camera_matrix
