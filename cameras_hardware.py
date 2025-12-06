import cv2
from cv2_enumerate_cameras import enumerate_cameras

cameras_detected = enumerate_cameras()


def get_available_cameras():
    cam_info = []
    for i, camera_info in enumerate(cameras_detected):
        status = "Inactive"
        try:
            test_cap = cv2.VideoCapture(camera_info.index)
            if test_cap.isOpened():
                status = "Inactive"
                test_cap.release()
            else:
                status = "Error"
        except:
            status = "Error"

        cam_info.append(
            {
                "id": camera_info.index,
                "name": camera_info.name,
                "index": camera_info.index,
                "path": camera_info.path if hasattr(camera_info, "path") else "N/A",
                "status": status,
            }
        )
    return cam_info


class camera_usb:
    count = 0

    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise Exception(f"Failed to open camera at index {index}")

    def get_id(self):
        """Return the camera index/ID"""
        return self.index

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def get_backend(self):
        return self.cap.getBackendName()

    def get_status(self):
        ret, _ = self.cap.read()
        return "Active" if ret else "Error"

    def get_resolution(self):
        width = self.get_width()
        height = self.get_height()
        return f"{width}x{height}"

    def set_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_fps(self, fps):
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def disable(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def take_picture(self):
        name = f"calibration_picture/picture_{self.count}.jpg"
        frame = self.get_frame()
        if frame is not None:
            cv2.imwrite(name, frame)
            self.count += 1
            return name
        return None


class camera_mipi_csi:
    def __init__(self) -> None:
        pass
