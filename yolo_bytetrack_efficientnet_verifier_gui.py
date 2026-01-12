import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
from collections import Counter
import os

# =====================
# GUI CHECK (AMAN)
# =====================
def is_gui_available():
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        return True
    except:
        return False

USE_GUI = is_gui_available()

# =====================
# CONFIG
# =====================
YOLO_MODEL_PATH = "model/best.pt"
VERIFIER_MODEL_PATH = "model/efficientnet_lite_verifier.pth"
VIDEO_PATH = "Data/IMG_8988.MOV"

TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

IMG_SIZE = 960
CONF_THRESHOLD = 0.3

SNAPSHOT_INTERVAL = 30      # HARUS SAMA DENGAN YOLO & BYTE TRACK
MIN_TRACK_AGE = 8
VERIFY_EVERY_N_FRAME = 4
VERIFIER_THRESHOLD = 0.35

WINDOW_NAME = "YOLO + ByteTrack + EfficientNet"
WINDOW_SIZE = 900

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# SIMPAN GAMBAR
# =====================
MAX_SAVED_IMAGES = 5
OUTPUT_DIR = "hasil_gambar/multimodel"
os.makedirs(OUTPUT_DIR, exist_ok=True)
saved_images = 0

# =====================
# LOAD MODEL
# =====================
yolo = YOLO(YOLO_MODEL_PATH)

verifier = models.efficientnet_b0(weights=None)
verifier.classifier[1] = nn.Linear(
    verifier.classifier[1].in_features, 2
)
verifier.load_state_dict(
    torch.load(VERIFIER_MODEL_PATH, map_location=DEVICE)
)
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

# =====================
# VIDEO
# =====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

if USE_GUI:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)
else:
    print("⚠️ GUI OpenCV tidak tersedia, berjalan headless")

# =====================
# STORAGE
# =====================
track_age = Counter()
snapshot_counts = []
frame_idx = 0

print("▶ Memproses video (YOLO + ByteTrack + EfficientNet-Lite)...")

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]

    results = yolo.track(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        tracker=TRACKER_PATH,
        persist=True,
        verbose=False
    )

    active_ids = set()
    annotated = frame.copy()

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            tid = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_age[tid] += 1
            if track_age[tid] < MIN_TRACK_AGE:
                continue

            # =====================
            # VERIFIER (PERIODIK)
            # =====================
            if frame_idx % VERIFY_EVERY_N_FRAME == 0:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                img_t = verifier_tf(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = verifier(img_t)
                    prob_valid = torch.softmax(out, dim=1)[0][1].item()

                if prob_valid < VERIFIER_THRESHOLD:
                    continue

            active_ids.add(tid)

            # DRAW
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"ID {tid}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # =====================
    # SNAPSHOT & SIMPAN GAMBAR
    # =====================
    if frame_idx % SNAPSHOT_INTERVAL == 0:
        snapshot_counts.append(len(active_ids))

        if saved_images < MAX_SAVED_IMAGES:
            save_path = os.path.join(
                OUTPUT_DIR,
                f"multimodel_snapshot_{frame_idx}.jpg"
            )
            if cv2.imwrite(save_path, annotated):
                saved_images += 1
                print(f"[INFO] Snapshot {frame_idx} disimpan → {save_path}")

    # =====================
    # GUI DISPLAY
    # =====================
    if USE_GUI:
        scale = min(WINDOW_SIZE / w, WINDOW_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(annotated, (new_w, new_h))

        canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        x_off = (WINDOW_SIZE - new_w) // 2
        y_off = (WINDOW_SIZE - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        cv2.putText(
            canvas,
            f"Benur aktif (terverifikasi): {len(active_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.imshow(WINDOW_NAME, canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    if frame_idx % 120 == 0:
        print(f"  Frame {frame_idx} diproses...")

cap.release()
if USE_GUI:
    cv2.destroyAllWindows()

# =====================
# FINAL ESTIMATION
# =====================
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
    f"melalui skema verifikasi periodik, "
    f"jumlah benur dalam video diestimasi berada pada kisaran "
    f"{mode_est}–{median_est} ekor."
)

print("\nProgram selesai dengan aman.")