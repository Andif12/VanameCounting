import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
from collections import Counter

# CONFIG (FINAL)
YOLO_MODEL_PATH = "model/best.pt"
VERIFIER_MODEL_PATH = "model/efficientnet_lite_verifier.pth"
VIDEO_PATH = "Data/objek 75.mp4"

TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

IMG_SIZE = 960
CONF_THRESHOLD = 0.3

SNAPSHOT_INTERVAL = 30          # snapshot tiap 30 frame
MIN_TRACK_AGE = 8               # ID harus stabil
VERIFY_EVERY_N_FRAME = 4        # verifier periodik
VERIFIER_THRESHOLD = 0.35       # 🔴 UBAH INI UNTUK EKSPERIMEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD YOLO
yolo = YOLO(YOLO_MODEL_PATH)

# LOAD VERIFIER
verifier = models.efficientnet_b0(weights=None)
verifier.classifier[1] = nn.Linear(verifier.classifier[1].in_features, 2)
verifier.load_state_dict(torch.load(VERIFIER_MODEL_PATH, map_location=DEVICE))
verifier.eval().to(DEVICE)

verifier_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

track_age = Counter()
snapshot_counts = []
frame_idx = 0

print("▶ Memproses video (FINAL MODE)...")

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = yolo.track(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        tracker=TRACKER_PATH,
        persist=True,
        verbose=False
    )

    active_ids = set()

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            tid = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

        
            # TRACK AGE UPDATE
        
            track_age[tid] += 1

            if track_age[tid] < MIN_TRACK_AGE:
                continue

        
            # VERIFIER (PERIODIK, NON HARD)
        
            if frame_idx % VERIFY_EVERY_N_FRAME == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                img_t = verifier_tf(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = verifier(img_t)
                    prob_valid = torch.softmax(out, dim=1)[0][1].item()

                if prob_valid < VERIFIER_THRESHOLD:
                    continue  # gugur hanya di frame ini

        
            # VALID BENUR
        
            active_ids.add(tid)


    # SNAPSHOT COUNT

    if frame_idx % SNAPSHOT_INTERVAL == 0:
        snapshot_counts.append(len(active_ids))

    if frame_idx % 120 == 0:
        print(f"  Frame {frame_idx} diproses...")

cap.release()

# FINAL ESTIMATION
if len(snapshot_counts) == 0:
    raise RuntimeError("Tidak ada snapshot yang berhasil diambil")

counter = Counter(snapshot_counts)
mode_est = counter.most_common(1)[0][0]
median_est = int(np.median(snapshot_counts))

print("\n===== HASIL ESTIMASI JUMLAH BENUR =====")
print(f"Total frame diproses            : {frame_idx}")
print(f"Total snapshot dianalisis       : {len(snapshot_counts)}")
print(f"Rentang hasil snapshot          : {min(snapshot_counts)} – {max(snapshot_counts)} benur")
print(f"Nilai modus (paling sering)     : {mode_est} benur")
print(f"Nilai median (tengah distribusi): {median_est} benur")

print("\n===== KESIMPULAN =====")
print(
    f"Dengan integrasi YOLO, ByteTrack, dan EfficientNet-Lite "
    f"dalam skema verifikasi periodik non-hard filtering, "
    f"jumlah benur dalam video diestimasi berada pada kisaran "
    f"{mode_est}–{median_est} ekor."
)

print("\nProgram selesai dengan aman.")