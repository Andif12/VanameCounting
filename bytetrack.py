from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter, defaultdict
import statistics
import math
import time
import os

# ================= CONFIG =================
MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/50.MP4"
TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

WINDOW_NAME = "YOLO + ByteTrack (Tiled 2x1)"
WINDOW_SIZE = 900

GROUND_TRUTH = 50

IMG_SIZE = 960
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.6
MAX_DET = 2000

TILE_ROWS = 2
TILE_COLS = 1

MAX_FRAMES = 300
BOX_THICKNESS = 1
BOX_ALPHA = 0.6

# ===== SIMPAN GAMBAR =====
SAVE_DIR = "hasil_deteksi/bytetrack/Objek 50"
SAVE_FRAME_INDEX = 300
TARGET_COUNT = 54   # simpan saat deteksi ini muncul
os.makedirs(SAVE_DIR, exist_ok=True)

saved_target = False

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)

# ================= STORAGE =================
counts_per_frame = []
frame_freq = defaultdict(int)

absolute_errors = []
squared_errors = []
percentage_errors = []

tp_total = 0
fp_total = 0
fn_total = 0

frame_idx = 0
start_time = time.time()

# ================= MAIN LOOP =================
while True:
    if frame_idx >= MAX_FRAMES:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    annotated = frame.copy()
    overlay = annotated.copy()

    h, w = frame.shape[:2]
    tile_h = h // TILE_ROWS
    tile_w = w // TILE_COLS

    all_boxes = []

    # ==== TILE + BYTETRACK ====
    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            y1 = i * tile_h
            y2 = (i + 1) * tile_h
            x1 = j * tile_w
            x2 = (j + 1) * tile_w

            tile = frame[y1:y2, x1:x2]

            results = model.track(
                tile,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                iou=IOU_THRESHOLD,
                max_det=MAX_DET,
                agnostic_nms=True,
                tracker=TRACKER_PATH,
                persist=False,
                verbose=False
            )

            if results and results[0].boxes is not None:
                for box in results[0].boxes.xyxy:
                    bx1, by1, bx2, by2 = map(int, box)
                    all_boxes.append(
                        (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                    )

    # ==== DRAW BOX ====
    for (x1, y1, x2, y2) in all_boxes:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            BOX_THICKNESS,
            lineType=cv2.LINE_AA
        )

    annotated = cv2.addWeighted(
        overlay, BOX_ALPHA, annotated, 1 - BOX_ALPHA, 0
    )

    detected_count = len(all_boxes)

    # ==== SIMPAN FRAME KE-300 ====
    if frame_idx == SAVE_FRAME_INDEX:
        path = f"{SAVE_DIR}/frame_{frame_idx}_pred_{detected_count}.jpg"
        cv2.imwrite(path, annotated)
        print(f"[SAVED] {path}")

    # ==== SIMPAN SAAT DETEKSI TARGET (637) ====
    if detected_count == TARGET_COUNT and not saved_target:
        path = f"{SAVE_DIR}/deteksi_{TARGET_COUNT}_frame_{frame_idx}.jpg"
        cv2.imwrite(path, annotated)
        print(f"[SAVED TARGET] {path}")
        saved_target = True

    # ==== COUNTING ====
    counts_per_frame.append(detected_count)
    frame_freq[detected_count] += 1

    error = detected_count - GROUND_TRUTH
    absolute_errors.append(abs(error))
    squared_errors.append(error ** 2)
    percentage_errors.append(abs(error) / GROUND_TRUTH)

    tp = min(detected_count, GROUND_TRUTH)
    fp = max(detected_count - GROUND_TRUTH, 0)
    fn = max(GROUND_TRUTH - detected_count, 0)

    tp_total += tp
    fp_total += fp
    fn_total += fn

    # ==== DISPLAY ====
    scale = min(WINDOW_SIZE / w, WINDOW_SIZE / h)
    resized = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    y_off = (WINDOW_SIZE - resized.shape[0]) // 2
    x_off = (WINDOW_SIZE - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

    cv2.putText(canvas, f"Benur (frame ini): {detected_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 0), 2)

    cv2.putText(canvas, f"Frame: {frame_idx}/{MAX_FRAMES}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 200, 200), 2)

    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# ================= FINAL =================
cap.release()
cv2.destroyAllWindows()
end_time = time.time()

mode_count = max(frame_freq, key=frame_freq.get)
median_count = int(statistics.median(counts_per_frame))
estimated_true_count = median_count

precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0
recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0
f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

mae = sum(absolute_errors) / len(absolute_errors)
rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
mape = (sum(percentage_errors) / len(percentage_errors)) * 100

std_dev = np.std(counts_per_frame)
cv = (std_dev / estimated_true_count) * 100 if estimated_true_count > 0 else 0

fps = len(counts_per_frame) / (end_time - start_time)

print("\n===== ESTIMASI JUMLAH BENUR =====")
print(f"Modus                        : {mode_count}")
print(f"Median                       : {median_count}")

print("\n===== METRIK KINERJA =====")
print(f"Precision                    : {precision:.4f}")
print(f"Recall                       : {recall:.4f}")
print(f"F1-Score                     : {f1_score:.4f}")
print(f"MAE                          : {mae:.2f}")
print(f"RMSE                         : {rmse:.2f}")
print(f"MAPE                         : {mape:.2f}%")
print(f"FPS                          : {fps:.2f}")