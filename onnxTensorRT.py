import time

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Load YOLO11n model
model = YOLO("content/jetson_orinnano.engine")

# Load image
img = Image.open("content/img.jpg").convert("RGB")
img = img.resize((640, 640))
img_tensor = transforms.ToTensor()(img).unsqueeze(0)

# Benchmark
num_iterations = 1000
total_time = 0.0

with torch.no_grad():
    for i in range(num_iterations):
        start_time = time.time()
        results = model(img_tensor)
        end_time = time.time()
        total_time += end_time - start_time

pytorch_fps = num_iterations / total_time
print(f"YOLO11n FPS: {pytorch_fps:.2f}")
