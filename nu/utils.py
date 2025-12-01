from math import atan

import cv2


def calc_yaw(bbox, f_x, img_width=640):
    x1, y1, x2, y2 = bbox.xyxy[0]
    x = (x1 + x2) / 2
    x -= img_width
    return atan(x / f_x)


def encode_video(frame):
    ret, buffer = cv2.imencode("jpg", frame)
    frame_bytes = buffer.tobytes()
    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
