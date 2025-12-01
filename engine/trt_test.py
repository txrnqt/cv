import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_PATH = "content/img.jpg"
NUM_RUNS = 100
print("=" * 70)
print("YOLO11n FPS Test")
print("=" * 70)

print("\nLoading YOLO11n model...")
model = YOLO("content/jetson_orinnano.engine")
print(f"Loading image: {IMAGE_PATH}")
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

print(f"Image size: {image.shape[1]}x{image.shape[0]}")

print("\nWarming up (5 runs)...")
for i in range(5):
    _ = model(image)

print(f"\nRunning {NUM_RUNS} inferences...")
inference_times = deque(maxlen=NUM_RUNS)

for i in range(NUM_RUNS):
    start = time.time()
    results = model(image, verbose=False)
    inference_time = (time.time() - start) * 1000
    inference_times.append(inference_time)

    if (i + 1) % 20 == 0:
        avg_time = np.mean(inference_times)
        avg_fps = 1000.0 / avg_time
        print(
            f"  Progress: {i + 1}/{NUM_RUNS} | Avg: {avg_time:.2f}ms | FPS: {avg_fps:.1f}"
        )

avg_inference = np.mean(inference_times)
min_inference = np.min(inference_times)
max_inference = np.max(inference_times)
avg_fps = 1000.0 / avg_inference

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Average Inference Time: {avg_inference:.2f}ms")
print(f"Min Inference Time:     {min_inference:.2f}ms")
print(f"Max Inference Time:     {max_inference:.2f}ms")
print(f"Average FPS:            {avg_fps:.1f}")
print("=" * 70)

print("\nRunning final detection with visualization")
results = model(image)
annotated = results[0].plot()

cv2.putText(
    annotated,
    f"Avg FPS: {avg_fps:.1f}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)
cv2.putText(
    annotated,
    f"Avg Inference: {avg_inference:.1f}ms",
    (10, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)

cv2.imshow("result", annotated)

print(f"\nDetected {len(results[0].boxes)} objects:")
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"  {model.names[cls]}: {conf:.3f}")
