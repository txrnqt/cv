import os
import threading

import cameras_hardware
import detect_nu
import utils
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

app = Flask(__name__)

latest_frame = None
latest_bbox = None
latest_yaw = None
frame_lock = threading.Lock()

camera = cameras_hardware.camera_usb(0)
detector = detect_nu.Detector("content/jetson_orinnano.engine")
camera_matrix = utils.get_calibration_camera_matrix("content/calibration.json")


def capture_loop():
    global latest_frame, latest_bbox, latest_yaw

    while True:
        frame = camera.get_frame()
        bbox = detector.bbox_frame(frame)
        yaw = utils.calculate_yaw(bbox, camera_matrix[0][0], camera.get_width())

        with frame_lock:
            latest_frame = frame.copy() if frame is not None else None
            latest_bbox = bbox.copy() if bbox is not None else None
            latest_yaw = yaw


def generate_frames(frame):
    while True:
        with frame_lock:
            if frame is not None:
                frame_generated = frame.copy()
            else:
                continue

        yield utils.encode_video(frame_generated)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera_matching/page")
def cameras_matching():
    return render_template("cameras.html")


@app.route("/config/page")
def configurator():
    return render_template("configurator.html")


@app.route("/video/raw_feed")
def video_feed():
    return Response(
        generate_frames(latest_frame), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video/detected_feed")
def detected_feed():
    return Response(
        generate_frames(latest_bbox),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detections/yaw")
def yaw():
    with frame_lock:
        current_yaw = latest_yaw
    return jsonify({"yaw": current_yaw})


@app.route("/config/upload_json", methods=["POST"])
def upload_json():
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            file.save(os.path.join("content/", file.filename))
            return "File uploaded successfully"
    return "No file selected"


def main():
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
