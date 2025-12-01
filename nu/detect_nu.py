import cv2
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)
        return results

    def bbox_frame(self, img):
        results = self.detect(img)
        filter = results[0]
        boxes = filter.boxes
        conf_thresh = boxes.conf > 0.75
        filter.boxes = boxes[conf_thresh]
        return filter
