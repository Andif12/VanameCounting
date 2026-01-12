import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
from collections import Counter
import statistics
import time

YOLO_MODEL = "model/best.pt"
VERIFIER_MODEL = "model/efficientnet_lite_verifier.pth"
VIDEO_PATH = "Data/IMG_8986.MOV"
TRACKER = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

IMG_SIZE = 960
CONF_THRESHOLD = 0.3
VERIFY_EVERY = 4
VERIFIER_THRESHOLD = 0.35
MIN_TRACK_AGE = 8
SNAPSHOT_INTERVAL = 30
MAX_FRAMES = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo = YOLO(YOLO_MODEL)
verifier = models.efficientnet_b0(weights=None)
verifier.classifier[1] = nn.Linear(verifier.classifier[1].in_features, 2)
verifier.load_state_dict(torch.load(VERIFIER_MODEL, map_location=device))
verifier.to(device).eval()

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

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
    result = yolo.track(
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
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            track_age[tid] += 1
            if track_age[tid] < MIN_TRACK_AGE:
                continue

            if frame_count % VERIFY_EVERY == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                img = tf(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.softmax(verifier(img), dim=1)[0][1].item()
                if prob < VERIFIER_THRESHOLD:
                    continue

            active_ids.add(tid)

    if frame_count % SNAPSHOT_INTERVAL == 0:
        snapshots.append(len(active_ids))

cap.release()
end = time.time()

median_val = statistics.median(snapshots)
std_val = np.std(snapshots)
cv_val = std_val / np.mean(snapshots)
fps = frame_count / (end - start)
time_pf = (end - start) / frame_count * 1000

print("\n===== EVALUASI MULTI-MODEL =====")
print(f"Total frame        : {frame_count}")
print(f"Median snapshot    : {median_val}")
print(f"Std Dev            : {std_val:.2f}")
print(f"CV                 : {cv_val:.3f}")
print(f"FPS                : {fps:.2f}")
print(f"Waktu / frame      : {time_pf:.2f} ms")
