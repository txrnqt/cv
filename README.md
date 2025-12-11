cv (i need a better name) is tensorrt infrence runner built for frc.

cv uses tensorrt yolo models in conjunction with the ultrylitics api to create a live cuda object detect pipeline. 
cv comes with a webserver to allow easy configuration of hardware. camera calibration is done with calibdb.net.

- engine
  - export.py - pytorch to engine yolo model translator
  - trt_test.py - test functinality of engine yolo models
- template - holds html files for flask
  - index.html - home page and hold camera feed for raw and processed camera feed
  - configurator.html - webpage for web configurator for the jetson
  - cameras.html - a camera matching tab
- cameras_hardware.py - hardware class for cameras as of now just usb cameras
- detec.py - detctor class for running yolo models
- main.py - contains webserver and runs the detection modles
- utils.py - contains useful methods such as calibraton json parser and object yaw calculator
