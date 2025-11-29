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
