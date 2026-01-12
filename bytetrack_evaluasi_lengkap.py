from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import os

# =====================
# CONFIG
# =====================
MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/IMG_8986.MOV"

TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

WINDOW_NAME = "YOLO + ByteTrack (Benur)"
WINDOW_SIZE = 900

IMG_SIZE = 1280
CONF_THRESHOLD = 0.3

SNAPSHOT_INTERVAL = 30      # HARUS sama dengan YOLO
MIN_VALID_ID = 8            # ID dianggap stabil jika muncul >= N frame

# SIMPAN GAMBAR
MAX_SAVED_IMAGES = 5
OUTPUT_DIR = "hasil_gambar/bytetrack_visual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# LOAD MODEL & VIDEO
# =====================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)

# =====================
# STORAGE
# =====================
frame_idx = 0
snapshot_counts = []
track_life = Counter()
saved_images = 0

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]

    results = model.track(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        tracker=TRACKER_PATH,
        persist=True,
        verbose=False
    )

    annotated = frame.copy()
    active_ids = set()

    # =====================
    # PROSES TRACKING
    # =====================
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            tid = int(box.id[0])
            track_life[tid] += 1

            if track_life[tid] >= MIN_VALID_ID:
                active_ids.add(tid)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # HIJAU = ID STABIL, ORANYE = BELUM STABIL
            color = (0, 255, 0) if tid in active_ids else (0, 165, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"ID {tid}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # =====================
    # SNAPSHOT COUNTING
    # =====================
    if frame_idx % SNAPSHOT_INTERVAL == 0:
        snapshot_counts.append(len(active_ids))

        # =====================
        # SIMPAN GAMBAR (FRAME SAMA DENGAN YOLO)
        # =====================
        if saved_images < MAX_SAVED_IMAGES:
            save_path = os.path.join(
                OUTPUT_DIR,
                f"bytetrack_snapshot_{frame_idx}.jpg"
            )

            success = cv2.imwrite(save_path, annotated)
            if success:
                saved_images += 1
                print(f"[INFO] Snapshot frame {frame_idx} disimpan → {save_path}")
            else:
                print(f"[WARNING] Gagal menyimpan frame {frame_idx}")

    # =====================
    # DISPLAY
    # =====================
    scale = min(WINDOW_SIZE / w, WINDOW_SIZE / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(annotated, (new_w, new_h))
    canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

    x_off = (WINDOW_SIZE - new_w) // 2
    y_off = (WINDOW_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    cv2.putText(
        canvas,
        f"Snapshot benur (ID stabil): {len(active_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

# =====================
# ANALISIS AKHIR
# =====================
mode_snapshot = Counter(snapshot_counts).most_common(1)[0][0]
median_snapshot = int(np.median(snapshot_counts))
min_snapshot = min(snapshot_counts)
max_snapshot = max(snapshot_counts)

print("\n===== HASIL ESTIMASI JUMLAH BENUR =====")
print(f"Total snapshot dianalisis        : {len(snapshot_counts)}")
print(f"Rentang hasil snapshot           : {min_snapshot} – {max_snapshot} benur")
print(f"Nilai modus (paling sering)      : {mode_snapshot} benur")
print(f"Nilai median (tengah distribusi) : {median_snapshot} benur")

print("\n===== KESIMPULAN ESTIMASI =====")
print(
    f"Berdasarkan snapshot counting berbasis ID stabil, "
    f"jumlah benur diestimasi berada pada rentang {min_snapshot}–{max_snapshot} ekor, "
    f"dengan nilai representatif sekitar {median_snapshot} ekor."
)

print("\nProgram selesai dengan aman.")
