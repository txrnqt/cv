import cv2
from flask import Flask, Response, jsonify, render_template

import cameras
import detect_nu
import utils

app = Flask(__name__)
camera = cameras.camera_usb(0)
detector = detect_nu.Detector("content/jetson_orinnano.engine")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        utils.encode_video(frame), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detect_video_feed")
def detect_video_feed():
    return Response(
        utils.encode_video(bbox),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/yaw")
def yaw():
    return Response(yaw)


while True:
    frame = camera.get_frame()
    bbox = detector.bbox_frame(frame)
    yaw = utils.calculate_yaw(bbox, asdf, camera.get_width())
