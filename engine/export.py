import tensorrt as trt
from ultralytics import YOLO

# Load your trained model
model = YOLO("content/jetson_orinnano.pt")

# Export to TensorRT engine
model.export(format="engine")
