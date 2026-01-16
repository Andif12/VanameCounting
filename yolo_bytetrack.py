from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter, defaultdict
import statistics
import math
import time

# ================= CONFIG =================
MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/300.mp4"
TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"

WINDOW_NAME = "YOLO + ByteTrack (Tiled 2x1)"
WINDOW_SIZE = 900

GROUND_TRUTH = 300   # jumlah manual sebenarnya

IMG_SIZE = 960
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.6
MAX_DET = 2000

TILE_ROWS = 2
TILE_COLS = 1

MAX_FRAMES = 300
BOX_THICKNESS = 1
BOX_ALPHA = 0.6

# Load model
model = YOLO(MODEL_PATH)

# Open video
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

    # ==== DRAW SMOOTH BOX ====
    for (x1, y1, x2, y2) in all_boxes:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),  # oranye = YOLO + ByteTrack
            BOX_THICKNESS,
            lineType=cv2.LINE_AA
        )

    annotated = cv2.addWeighted(
        overlay,
        BOX_ALPHA,
        annotated,
        1 - BOX_ALPHA,
        0
    )

    # ==== COUNTING ====
    detected_count = len(all_boxes)
    counts_per_frame.append(detected_count)
    frame_freq[detected_count] += 1

    # ==== ERROR PER FRAME ====
    error = detected_count - GROUND_TRUTH
    absolute_errors.append(abs(error))
    squared_errors.append(error ** 2)
    percentage_errors.append(abs(error) / GROUND_TRUTH)

    # ==== COUNTING-BASED PRECISION / RECALL ====
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

    cv2.putText(
        canvas,
        f"Benur (frame ini): {detected_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Frame: {frame_idx}/{MAX_FRAMES}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2
    )

    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
end_time = time.time()

# ================= ANALISIS AKHIR =================
mode_count = max(frame_freq, key=frame_freq.get)
median_count = int(statistics.median(counts_per_frame))
estimated_true_count = median_count

precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

mae = sum(absolute_errors) / len(absolute_errors)
rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
mape = (sum(percentage_errors) / len(percentage_errors)) * 100

std_dev = np.std(counts_per_frame)
cv = (std_dev / estimated_true_count) * 100 if estimated_true_count > 0 else 0

abs_error = abs(estimated_true_count - GROUND_TRUTH)
percentage_error = (abs_error / GROUND_TRUTH) * 100

total_time = end_time - start_time
fps = len(counts_per_frame) / total_time
time_per_frame = total_time / len(counts_per_frame)

# ================= OUTPUT =================
print("\n===== ESTIMASI JUMLAH BENUR =====")
print(f"Modus                        : {mode_count}")
print(f"Median                       : {median_count}")

print("\n===== JUMLAH OBJEK DIANGGAP BENAR =====")
print(f"Estimasi akhir (median)      : {estimated_true_count}")

print("\n===== METRIK KINERJA (COUNTING-BASED) =====")
print(f"Precision                    : {precision:.4f}")
print(f"Recall                       : {recall:.4f}")
print(f"F1-Score                     : {f1_score:.4f}")

print("\n===== ERROR TERHADAP GROUND TRUTH =====")
print(f"Selisih absolut              : {abs_error}")
print(f"Error persentase             : {percentage_error:.2f}%")

print("\n===== ERROR METRICS PER FRAME =====")
print(f"MAE                          : {mae:.2f}")
print(f"RMSE                         : {rmse:.2f}")
print(f"MAPE                         : {mape:.2f}%")

print("\n===== STATISTIK & PERFORMA =====")
print(f"Standard Deviation           : {std_dev:.2f}")
print(f"Coefficient of Variation     : {cv:.2f}%")
print(f"FPS                          : {fps:.2f}")
print(f"Waktu / frame                : {time_per_frame:.4f} detik")

print("\nProgram selesai dengan aman.")