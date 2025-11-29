from cv2_enumerate_cameras import enumerate_cameras

cameras = enumerate_cameras()

if cameras:
    print("Available Cameras:")
    for i, camera_info in enumerate(cameras):
        print(f"Camera {i}:")
        print(f"  Name: {camera_info.name}")
        print(f"  OpenCV Index: {camera_info.index}")
        print(f"  Vendor ID: {camera_info.vid}")
        print(f"  Product ID: {camera_info.pid}")
        print("-" * 20)
else:
    print("No cameras found.")
