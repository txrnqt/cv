import os
import threading
import time

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
)

import cameras_hardware
import detect
import utils

app = Flask(__name__)
latest_frame = None
latest_annotated_frame = None
latest_results = None
latest_yaw = None
frame_lock = threading.Lock()
has_json = False
camera_matrix = None
calibration_path = "content/calibration.json"
cameras = {}

detector = detect.Detector("content/jetson_orinnano.engine")


if os.path.exists(calibration_path):
    try:
        camera_matrix = utils.get_calibration_camera_matrix(calibration_path)
        print(f"Loaded calibration matrix from {calibration_path}")
    except Exception as e:
        print(f"Failed to load calibration: {e}")
        camera_matrix = None
else:
    print(f"Calibration file not found at {calibration_path}")


def capture_loop():
    global latest_frame, latest_annotated_frame, latest_results, latest_yaw
    while True:
        try:
            frame = camera.get_frame()
            results = detector.bbox_frame(frame)
            annotated_frame = None
            if results is not None:
                annotated_frame = results.plot()
            if has_json:
                yaw = utils.calculate_yaw(results, camera_matrix, camera.get_width())
            else:
                yaw = 1
            with frame_lock:
                latest_frame = frame.copy() if frame is not None else None
                latest_annotated_frame = annotated_frame
                latest_results = results
                latest_yaw = yaw
        except Exception as e:
            print(f"Error in capture loop: {e}")


def generate_frames(frame_type="raw"):
    last_frame = None
    frame_interval = 1.0 / 30  # 30 FPS max
    last_time = 0

    while True:
        current_time = time.time()

        # Rate limiting
        if current_time - last_time < frame_interval:
            time.sleep(0.01)
            continue

        last_time = current_time

        with frame_lock:
            if frame_type == "raw":
                frame = latest_frame
            else:  # detected
                frame = latest_annotated_frame
            if frame is None:
                time.sleep(0.033)
                continue

            frame_generated = frame.copy()

        ret, buffer = cv2.imencode(
            ".jpg", frame_generated, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera_matching/page")
def cameras_matching_page():
    return render_template("cameras.html")


@app.route("/config/page")
def configurator_page():
    return render_template("configurator.html")


@app.route("/video/raw_feed")
def video_feed_raw():
    return Response(
        generate_frames("raw"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video/detected_feed")
def video_feed_processed():
    return Response(
        generate_frames("detected"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detections/yaw")
def yaw():
    with frame_lock:
        current_yaw = latest_yaw
    return jsonify({"yaw": current_yaw})


@app.route("/config/upload_json", methods=["POST"])
def upload_json():
    global camera_matrix

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".json"):
        return jsonify({"error": "File must be a JSON file"}), 400

    # Save the file
    filepath = os.path.join("content/", file.filename)
    file.save(filepath)

    # Try to load the calibration matrix
    try:
        camera_matrix = utils.get_calibration_camera_matrix(filepath)
        return jsonify(
            {
                "message": "File uploaded and calibration loaded successfully",
                "calibration_loaded": True,
            }
        ), 200
    except Exception as e:
        return jsonify(
            {
                "message": "File uploaded but failed to load calibration reupload",
                "error": str(e),
                "calibration_loaded": False,
            }
        ), 200


@app.route("/config/status")
def config_status():
    return jsonify(
        {
            "calibration_loaded": camera_matrix is not None,
            "calibration_path": calibration_path if camera_matrix is not None else None,
        }
    )


@app.route("/video/avaliable_cameras")
def avaliable_cameras_button():
    cameras_avaliable = cameras_hardware.get_available_cameras()
    return jsonify(cameras_avaliable)


@app.route("/video/activate_camera", methods=["POST"])
def activate_camera():
    global camera

    data = request.get_json()
    camera_id = data.get("camera_id", 0)

    camera = cameras_hardware.camera_usb(camera_id)
    cameras.append(camera)

    return {
        "success": True,
        "message": f"Camera {camera_id} opened successfully",
        "camera_id": camera_id,
    }


def main():
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
