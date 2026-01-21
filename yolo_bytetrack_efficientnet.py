from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import statistics
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import time
import sys
import math

# CONFIG
MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/300.MP4"
TRACKER_PATH = "venv/Lib/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"
VERIFIER_MODEL_PATH = "model/efficientnet_lite_verifier.pth"

WINDOW_NAME = "YOLO + ByteTrack + EfficientNet-Lite Verifier"
WINDOW_SIZE = 900

GROUND_TRUTH = 75

IMG_SIZE = 960
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.6
MAX_DET = 2000

TILE_ROWS = 2
TILE_COLS = 1
MAX_FRAMES = 300

# visual smoothing
BOX_THICKNESS = 1
BOX_ALPHA = 0.6

# verifier sampling
VERIFIER_INTERVAL = 5
MAX_VERIFY_PER_FRAME = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODELS
detector = YOLO(MODEL_PATH)

verifier = efficientnet_b0(weights=None)
verifier.classifier[1] = torch.nn.Linear(
    verifier.classifier[1].in_features, 2
)

state_dict = torch.load(VERIFIER_MODEL_PATH, map_location=DEVICE)
verifier.load_state_dict(state_dict)
verifier.to(DEVICE)
verifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# VIDEO
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)

# STORAGE
counts_per_frame = []
frame_freq = defaultdict(int)

tp_total = 0
fp_total = 0
fn_total = 0

frame_idx = 0
start_time = time.time()
force_stop = False

# MAIN LOOP
try:
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

        final_boxes = []

        # YOLO + BYTETRACK + VERIFIER
        for i in range(TILE_ROWS):
            for j in range(TILE_COLS):
                y1 = i * tile_h
                y2 = (i + 1) * tile_h
                x1 = j * tile_w
                x2 = (j + 1) * tile_w

                tile = frame[y1:y2, x1:x2]

                results = detector.track(
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
                    verify_count = 0

                    for box in results[0].boxes.xyxy:
                        bx1, by1, bx2, by2 = map(int, box)
                        is_valid = True

                        if (
                            frame_idx % VERIFIER_INTERVAL == 0
                            and verify_count < MAX_VERIFY_PER_FRAME
                        ):
                            crop = tile[by1:by2, bx1:bx2]
                            if crop.size != 0:
                                crop_pil = Image.fromarray(
                                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                )
                                input_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)

                                with torch.no_grad():
                                    pred = torch.argmax(
                                        verifier(input_tensor), dim=1
                                    ).item()

                                is_valid = (pred == 1)
                                verify_count += 1

                        if is_valid:
                            final_boxes.append(
                                (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                            )

        # DRAW
        for (x1, y1, x2, y2) in final_boxes:
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

        detected_count = len(final_boxes)
        counts_per_frame.append(detected_count)
        frame_freq[detected_count] += 1

        # COUNTING METRICS
        tp = min(detected_count, GROUND_TRUTH)
        fp = max(detected_count - GROUND_TRUTH, 0)
        fn = max(GROUND_TRUTH - detected_count, 0)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        # DISPLAY
        scale = min(WINDOW_SIZE / w, WINDOW_SIZE / h)
        resized = cv2.resize(annotated, (int(w * scale), int(h * scale)))

        canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
        y_off = (WINDOW_SIZE - resized.shape[0]) // 2
        x_off = (WINDOW_SIZE - resized.shape[1]) // 2
        canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

        cv2.putText(canvas, f"Benur terdeteksi: {detected_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(canvas, f"Frame: {frame_idx}/{MAX_FRAMES}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

end_time = time.time()

# ANALISIS
if len(counts_per_frame) == 0:
    print("Tidak ada frame yang berhasil diproses.")
    sys.exit()

mode_benur = max(frame_freq, key=frame_freq.get)
median_benur = int(statistics.median(counts_per_frame))
estimated_true_count = median_benur

precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

errors = [c - GROUND_TRUTH for c in counts_per_frame]
abs_errors = [abs(e) for e in errors]

mae = np.mean(abs_errors)
rmse = math.sqrt(np.mean([e**2 for e in errors]))
mape = np.mean([abs(e) / GROUND_TRUTH for e in errors]) * 100

std_dev = np.std(counts_per_frame)
cv = (std_dev / estimated_true_count) * 100 if estimated_true_count > 0 else 0

total_time = end_time - start_time
fps = len(counts_per_frame) / total_time
time_per_frame = total_time / len(counts_per_frame)

# OUTPUT
print("\nESTIMASI JUMLAH BENUR")
print(f"Modus                        : {mode_benur}")
print(f"Median                       : {median_benur}")

print("\nJUMLAH OBJEK DIANGGAP BENAR")
print(f"Estimasi akhir (median)      : {estimated_true_count}")

print("\nMETRIK KINERJA (COUNTING-BASED)")
print(f"Precision                    : {precision:.4f}")
print(f"Recall                       : {recall:.4f}")
print(f"F1-Score                     : {f1_score:.4f}")

print("\nERROR METRICS PER FRAME")
print(f"MAE                          : {mae:.2f}")
print(f"RMSE                         : {rmse:.2f}")
print(f"MAPE                         : {mape:.2f}%")

print("\nSTATISTIK DISTRIBUSI")
print(f"Standard Deviation           : {std_dev:.2f}")
print(f"Coefficient of Variation     : {cv:.2f}%")

print("\nPERFORMA SISTEM")
print(f"FPS                          : {fps:.2f}")
print(f"Waktu / frame                : {time_per_frame:.4f} detik")