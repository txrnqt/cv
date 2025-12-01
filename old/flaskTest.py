import cv2
from flask import Flask, Response, jsonify, render_template

import cameras as cam_data
import detect_main as detect

app = Flask(__name__)


def generate_frames():
    while True:
        frame = detect.get_frame_raw()
        if frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


def generate_frames_detected():
    while True:
        frame = detect.get_frame()
        if frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cameras_pages")
def cameras():
    return render_template("cameras.html")


@app.route("/video_feed")
def video_feed():
    print("video_feed route called")
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_feed_detected")
def video_feed_detected():
    print("video_feed_detected route called")
    return Response(
        generate_frames_detected(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/cameras")
def cameras_data():
    cameras = cam_data.get_available_cameras()
    return jsonify(cameras)


@app.route("/api/camera/<int:camera_id>/preview")
def camera_preview(camera_id):
    def generate():
        cameras_list = cam_data.get_available_cameras()
        if camera_id >= len(cameras_list):
            return

        camera_info = list(cameras_list)[camera_id]
        cap = cv2.VideoCapture(camera_info.index())

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    break

                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
        finally:
            cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start_server(host="0.0.0.0", port=5000, debug=False):
    app.run(debug=debug, host=host, port=port, threaded=True, use_reloader=False)


def release_camera():
    detect.release_camera()


if __name__ == "__main__":
    start_server(port=5000, debug=False)
    release_camera()
