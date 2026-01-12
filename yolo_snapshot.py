from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/IMG_8987.MOV"

IMG_SIZE = 1280
CONF_THRESHOLD = 0.3
SNAPSHOT_INTERVAL = 30   # ambil 1 frame tiap 30 frame

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

snapshot_counts = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if frame_idx % SNAPSHOT_INTERVAL != 0:
        continue

    results = model(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        rect=True
    )

    count = len(results[0].boxes)
    snapshot_counts.append(count)

cap.release()

# ==============================
# ANALISIS SNAPSHOT
# ==============================
mode_count = Counter(snapshot_counts).most_common(1)[0][0]
median_count = int(np.median(snapshot_counts))

print("\n===== HASIL SNAPSHOT COUNTING =====")
print(f"Total snapshot diambil : {len(snapshot_counts)}")
print(f"Mode snapshot          : {mode_count}")
print(f"Median snapshot        : {median_count}")
print(f"Snapshot counts        : {snapshot_counts}")
print("Program selesai dengan aman.")
