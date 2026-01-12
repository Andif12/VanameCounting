from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# CONFIG
MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/IMG_8988.MOV"
TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

WINDOW_NAME = "YOLO + ByteTrack (Benur)"
WINDOW_SIZE = 900

IMG_SIZE = 1280
CONF_THRESHOLD = 0.3

SNAPSHOT_INTERVAL = 30
MIN_VALID_ID = 8

# LOAD MODEL
model = YOLO(MODEL_PATH)

# LOAD VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)

# STORAGE
frame_idx = 0
snapshot_counts = []
track_life = Counter()

# MAIN LOOP
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
        tracker=TRACKER_PATH,   # ✅ PATH YAML RESMI
        persist=True,
        verbose=False
    )

    annotated = frame.copy()
    active_ids = set()

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue

            tid = int(box.id[0])
            track_life[tid] += 1

            if track_life[tid] >= MIN_VALID_ID:
                active_ids.add(tid)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
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


    # SNAPSHOT COUNTING

    if frame_idx % SNAPSHOT_INTERVAL == 0:
        snapshot_counts.append(len(active_ids))


    # DISPLAY

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
        f"Snapshot benur: {len(active_ids)}",
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

# ANALISIS AKHIR (ESTIMASI)
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
    f"Berdasarkan hasil snapshot counting, jumlah benur dalam video "
    f"diestimasi berada pada rentang {min_snapshot}–{max_snapshot} ekor, "
    f"dengan nilai representatif sekitar {median_snapshot} ekor."
)

print("\nProgram selesai dengan aman.")
