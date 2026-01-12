from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import statistics
import os
import csv

# CONFIG
MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/IMG_8988.MOV"

WINDOW_NAME = "YOLO Detection (Benur)"
WINDOW_SIZE = 900

IMG_SIZE = 1280
CONF_THRESHOLD = 0.4

# SIMPAN GAMBAR
SNAPSHOT_INTERVAL = 30          # tiap N frame
MAX_SAVED_IMAGES = 5            # maksimal gambar
OUTPUT_IMG_DIR = "hasil_gambar/yolo"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# SIMPAN DATA NUMERIK
OUTPUT_CSV = "hasil_gambar/yolo/yolo_counts_per_frame.csv"

# LOAD MODEL & VIDEO
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)

# STORAGE
frame_count_frequency = defaultdict(int)
counts_per_frame = []
total_frames = 0
saved_images = 0

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    orig_h, orig_w = frame.shape[:2]

    results = model(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        rect=True
    )

    annotated = frame.copy()


    # HITUNG BENUR FRAME INI

    current_benur_count = len(results[0].boxes)
    frame_count_frequency[current_benur_count] += 1
    counts_per_frame.append(current_benur_count)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # SIMPAN GAMBAR SNAPSHOT

    if total_frames % SNAPSHOT_INTERVAL == 0 and saved_images < MAX_SAVED_IMAGES:
        save_path = os.path.join(
            OUTPUT_IMG_DIR,
            f"yolo_frame_{total_frames}.jpg"
        )
        if cv2.imwrite(save_path, annotated):
            saved_images += 1
            print(f"[INFO] Gambar disimpan → {save_path}")


    # DISPLAY (LETTERBOX)

    scale = min(WINDOW_SIZE / orig_w, WINDOW_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(annotated, (new_w, new_h))
    canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

    x_offset = (WINDOW_SIZE - new_w) // 2
    y_offset = (WINDOW_SIZE - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    cv2.putText(
        canvas,
        f"Benur (frame ini): {current_benur_count}",
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

# ANALISIS TANPA GROUND TRUTH
mode_benur = max(frame_count_frequency, key=frame_count_frequency.get)
mode_freq = frame_count_frequency[mode_benur]
median_benur = statistics.median(counts_per_frame)
mean_benur = statistics.mean(counts_per_frame)
std_benur = statistics.pstdev(counts_per_frame)
cv_benur = std_benur / mean_benur if mean_benur > 0 else 0

# SIMPAN CSV (HASIL TAMBAHAN)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Jumlah_Benur"])
    for i, c in enumerate(counts_per_frame, start=1):
        writer.writerow([i, c])

# OUTPUT AKHIR VIDEO
print("\n===== HASIL AKHIR VIDEO (YOLO) =====")
print(f"Total frame diproses              : {total_frames}")

print("\nDistribusi jumlah benur / frame :")
for k in sorted(frame_count_frequency):
    print(f"  {k} benur → {frame_count_frequency[k]} frame")

print("\n===== ESTIMASI JUMLAH BENUR =====")
print(f"Modus (paling sering muncul)      : {mode_benur} benur")
print(f"Muncul pada                       : {mode_freq} frame")
print(f"Median (nilai tengah)             : {median_benur} benur")
print(f"Rata-rata                         : {mean_benur:.2f} benur")

print("\n===== STABILITAS =====")
print(f"Std Dev                           : {std_benur:.2f}")
print(f"Coefficient of Variation          : {cv_benur:.3f}")

print("\n===== KESIMPULAN =====")
print(
    f"Berdasarkan hasil deteksi YOLO tanpa pelacakan, "
    f"jumlah benur dalam video diestimasi sekitar {mode_benur} ekor "
    f"berdasarkan nilai modus yang paling sering muncul."
)

print("\nProgram selesai dengan aman.")