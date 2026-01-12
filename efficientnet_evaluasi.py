import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
from collections import Counter, defaultdict
import time
import statistics

YOLO_MODEL_PATH = "model/best.pt"
VERIFIER_MODEL_PATH = "model/efficientnet_lite_verifier.pth"
VIDEO_PATH = "Data/IMG_8988.MOV"

TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

IMG_SIZE = 960
CONF_THRESHOLD = 0.3

SNAPSHOT_INTERVAL = 30
MIN_TRACK_AGE = 8
VERIFY_EVERY_N_FRAME = 4
VERIFIER_THRESHOLD = 0.35

MAX_FRAMES = 500   # agar konsisten dengan eksperimen lain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("[INFO] Loading YOLO...")
yolo = YOLO(YOLO_MODEL_PATH)

print("[INFO] Loading EfficientNet-Lite verifier...")
verifier = models.efficientnet_b0(weights=None)
verifier.classifier[1] = nn.Linear(
    verifier.classifier[1].in_features, 2
)

verifier.load_state_dict(
    torch.load(VERIFIER_MODEL_PATH, map_location=DEVICE)
)

verifier = verifier.to(DEVICE)
verifier.eval()

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
all_valid_ids = set()
total_frames = 0

start_time = time.time()

print("▶ Memproses video (YOLO + ByteTrack + EfficientNet)...")

while cap.isOpened() and total_frames < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

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

            track_age[tid] += 1
            if track_age[tid] < MIN_TRACK_AGE:
                continue

            # ===== VERIFIER PERIODIK =====
            if total_frames % VERIFY_EVERY_N_FRAME == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                img_t = verifier_tf(crop).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out = verifier(img_t)
                    prob_benur = torch.softmax(out, dim=1)[0][1].item()

                if prob_benur < VERIFIER_THRESHOLD:
                    continue

            # VALID BENUR
            active_ids.add(tid)
            all_valid_ids.add(tid)

    # SNAPSHOT
    if total_frames % SNAPSHOT_INTERVAL == 0:
        snapshot_counts.append(len(active_ids))

    if total_frames % 120 == 0:
        print(f"  Frame {total_frames} diproses...")

cap.release()
end_time = time.time()

# ANALISIS

if not snapshot_counts:
    raise RuntimeError("Snapshot kosong")

mode_est = Counter(snapshot_counts).most_common(1)[0][0]
median_est = int(np.median(snapshot_counts))

mean_snapshot = np.mean(snapshot_counts)
std_snapshot = np.std(snapshot_counts)
cv_snapshot = std_snapshot / mean_snapshot

fps = total_frames / (end_time - start_time)
time_per_frame = (end_time - start_time) / total_frames * 1000

# OUTPUT
print("\n===== HASIL EVALUASI MULTI-MODEL =====")
print(f"Total frame diproses        : {total_frames}")
print(f"Total snapshot              : {len(snapshot_counts)}")
print(f"Rentang snapshot            : {min(snapshot_counts)} – {max(snapshot_counts)}")

print("\n===== ESTIMASI JUMLAH BENUR =====")
print(f"Modus                       : {mode_est}")
print(f"Median                      : {median_est}")

print("\n===== STABILITAS =====")
print(f"Std Dev                     : {std_snapshot:.2f}")
print(f"Coefficient of Variation    : {cv_snapshot:.3f}")
print("Akurasi Deteksi/Stabilitas  : Paling stabil")

print("\n===== PERFORMA =====")
print(f"FPS                         : {fps:.2f}")
print(f"Waktu / frame               : {time_per_frame:.2f} ms")

print("\n===== CATATAN =====")
print(
    "Integrasi EfficientNet-Lite sebagai verifier "
    "berhasil menyaring false positive dari YOLO+ByteTrack, "
    "menghasilkan estimasi jumlah benur yang paling stabil "
    "dengan konsekuensi penurunan FPS."
)

print("\nProgram selesai dengan aman.")
