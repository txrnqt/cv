import cv2


class usb_camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    def get_frame(self):
        s, img = self.cap.read()
        if s:
            pass
        return img

    def relase_camera(self):
        self.cap.release()
