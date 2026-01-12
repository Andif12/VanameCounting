import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import statistics
import time

MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/IMG_8986.MOV"
TRACKER = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

IMG_SIZE = 1280
CONF_THRESHOLD = 0.3
SNAPSHOT_INTERVAL = 30
MIN_TRACK_AGE = 8
MAX_FRAMES = 500

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

track_age = Counter()
snapshots = []
frame_count = 0
start = time.time()

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    result = model.track(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        tracker=TRACKER,
        persist=True,
        verbose=False
    )

    active_ids = set()
    if result[0].boxes is not None:
        for box in result[0].boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            track_age[tid] += 1
            if track_age[tid] >= MIN_TRACK_AGE:
                active_ids.add(tid)

    if frame_count % SNAPSHOT_INTERVAL == 0:
        snapshots.append(len(active_ids))

cap.release()
end = time.time()

median_val = statistics.median(snapshots)
std_val = np.std(snapshots)
cv_val = std_val / np.mean(snapshots)
mae = np.mean([abs(s - median_val) for s in snapshots])
mape = np.mean([abs(s - median_val) / median_val for s in snapshots]) * 100
fps = frame_count / (end - start)
time_pf = (end - start) / frame_count * 1000

print("\n===== EVALUASI YOLO + ByteTrack =====")
print(f"Total frame        : {frame_count}")
print(f"Median snapshot    : {median_val}")
print(f"Std Dev            : {std_val:.2f}")
print(f"CV                 : {cv_val:.3f}")
print(f"MAE                : {mae:.2f}")
print(f"MAPE               : {mape:.2f} %")
print(f"FPS                : {fps:.2f}")
print(f"Waktu / frame      : {time_pf:.2f} ms")
