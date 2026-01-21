from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import statistics
import math
import time

MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/300.MP4"

GROUND_TRUTH_BENUR = 75   # jumlah manual sebenarnya

WINDOW_NAME = "YOLO Detection (Tiled)"
WINDOW_SIZE = 900

IMG_SIZE = 960
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.6
MAX_DET = 2000

TILE_ROWS = 2
TILE_COLS = 2

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

frame_count_frequency = defaultdict(int)
counts_per_frame = []

absolute_errors = []
squared_errors = []
percentage_errors = []

tp_total = 0
fp_total = 0
fn_total = 0

frame_idx = 0
total_frames = 0

start_time = time.time()

while True:
    if frame_idx >= MAX_FRAMES:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    total_frames += 1

    annotated = frame.copy()
    overlay = annotated.copy()

    h, w = frame.shape[:2]
    tile_h = h // TILE_ROWS
    tile_w = w // TILE_COLS

    total_boxes = []

    for i in range(TILE_ROWS):
        for j in range(TILE_COLS):
            y1 = i * tile_h
            y2 = (i + 1) * tile_h
            x1 = j * tile_w
            x2 = (j + 1) * tile_w

            tile = frame[y1:y2, x1:x2]

            results = model(
                tile,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=MAX_DET,
                agnostic_nms=True,
                verbose=False
            )

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    total_boxes.append(
                        (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                    )

    for (x1, y1, x2, y2) in total_boxes:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),  
            BOX_THICKNESS,
            lineType=cv2.LINE_AA
        )

    annotated = cv2.addWeighted(
        overlay, BOX_ALPHA, annotated, 1 - BOX_ALPHA, 0
    )

    detected_count = len(total_boxes)
    counts_per_frame.append(detected_count)
    frame_count_frequency[detected_count] += 1

    error = detected_count - GROUND_TRUTH_BENUR
    absolute_errors.append(abs(error))
    squared_errors.append(error ** 2)
    percentage_errors.append(abs(error) / GROUND_TRUTH_BENUR)

    tp = min(detected_count, GROUND_TRUTH_BENUR)
    fp = max(detected_count - GROUND_TRUTH_BENUR, 0)
    fn = max(GROUND_TRUTH_BENUR - detected_count, 0)

    tp_total += tp
    fp_total += fp
    fn_total += fn

    #display
    scale = min(WINDOW_SIZE / w, WINDOW_SIZE / h)
    resized = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    y_off = (WINDOW_SIZE - resized.shape[0]) // 2
    x_off = (WINDOW_SIZE - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

    cv2.putText(canvas, f"Benur (frame ini): {detected_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(canvas, f"Frame: {frame_idx}/{MAX_FRAMES}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
end_time = time.time()

#ANALISIS AKHIR
mode_benur = max(frame_count_frequency, key=frame_count_frequency.get)
median_benur = int(statistics.median(counts_per_frame))
estimated_true_count = median_benur

precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
f1_score  = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

mae  = sum(absolute_errors) / total_frames
rmse = math.sqrt(sum(squared_errors) / total_frames)
mape = (sum(percentage_errors) / total_frames) * 100

std_dev = np.std(counts_per_frame)
cv = (std_dev / estimated_true_count) * 100 if estimated_true_count > 0 else 0

abs_error = abs(estimated_true_count - GROUND_TRUTH_BENUR)
percentage_error = (abs_error / GROUND_TRUTH_BENUR) * 100

total_time = end_time - start_time
fps = total_frames / total_time
time_per_frame = total_time / total_frames

# karakter hasil
if cv < 10:
    karakter = "Sangat stabil"
elif cv < 20:
    karakter = "Stabil"
else:
    karakter = "Fluktuatif"

#OUTPUT
print("\n===== HASIL AKHIR VIDEO =====")
print(f"Ground truth (manual)        : {GROUND_TRUTH_BENUR}")
print(f"Total frame diuji            : {total_frames}")

print("\n===== ESTIMASI JUMLAH BENUR =====")
print(f"Modus                        : {mode_benur}")
print(f"Median                       : {median_benur}")

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
print(f"Karakter hasil               : {karakter}")
