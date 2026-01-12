import cv2
import numpy as np
from ultralytics import YOLO
import statistics
import time

MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/IMG_8986.MOV"

IMG_SIZE = 1280
CONF_THRESHOLD = 0.3
MAX_FRAMES = 500

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

counts = []
start = time.time()
frame_count = 0

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    result = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
    counts.append(len(result[0].boxes))

cap.release()
end = time.time()

mean_val = np.mean(counts)
median_val = statistics.median(counts)
std_val = np.std(counts)
cv_val = std_val / mean_val
mae = np.mean([abs(c - median_val) for c in counts])
mape = np.mean([abs(c - median_val) / median_val for c in counts]) * 100
fps = frame_count / (end - start)
time_pf = (end - start) / frame_count * 1000

print("\n===== EVALUASI YOLO =====")
print(f"Total frame        : {frame_count}")
print(f"Rata-rata / frame  : {mean_val:.2f}")
print(f"Median             : {median_val}")
print(f"Std Dev            : {std_val:.2f}")
print(f"CV                 : {cv_val:.3f}")
print(f"MAE                : {mae:.2f}")
print(f"MAPE               : {mape:.2f} %")
print(f"FPS                : {fps:.2f}")
print(f"Waktu / frame      : {time_pf:.2f} ms")