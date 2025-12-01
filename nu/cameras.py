import cv2
from cv2_enumerate_cameras import enumerate_cameras

cameras = enumerate_cameras()


def get_available_cameras():
    cam_info = []
    for i, camera_info in enumerate(cameras):
        cap = cv2.VideoCapture(camera_info.index)

        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            backend = cap.getBackendName()

            ret, frame = cap.read()
            status = "Active" if ret else "Error"

            cap.release()
        else:
            width = height = fps = 0
            backend = "Unknown"
            status = "Inactive"

        cam_info.append(
            {
                "id": i,
                "name": camera_info.name,
                "index": camera_info.index,
                "path": camera_info.path if hasattr(camera_info, "path") else "N/A",
                "resolution": f"{width}x{height}",
                "fps": fps,
                "backend": backend,
                "status": status,
            }
        )
    return cam_info


class camera_usb:
    count = 0

    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(index)

    def get_frame(self):
        ret, frame = self.cap.read()
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
        self.cap.release()

    def take_picture(self):
        name = "calibration_picture/"
        cv2.imwrite(name, self.get_frame())
        self.count += 1


class camera_mipi_csi:
    def __init__(self) -> None:
        pass
