from ultralytics import YOLO
import cv2
import pandas as pd
import time

VIDEO_PATH = "Data/IMG_8988.MOV"
MODEL_PATH = "model/best.pt"
MAX_FRAMES = 500        # 🔴 batasi untuk eksperimen
CONFIDENCE = 0.25

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0
results_data = []

start_time = time.time()

while cap.isOpened() and frame_id < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False   # 🔴 MATIKAN LOG YOLO
    )

    jumlah_benur = len(results[0].boxes)
    results_data.append({
        "frame": frame_id,
        "jumlah_benur": jumlah_benur
    })

    frame_id += 1

cap.release()
end_time = time.time()

# Simpan CSV
df = pd.DataFrame(results_data)
df.to_csv("hasil_yolo_per_frame.csv", index=False)

print("=== SELESAI ===")
print("Total frame diproses:", frame_id)
print("Rata-rata benur/frame:", df["jumlah_benur"].mean())
print("Waktu total (detik):", round(end_time - start_time, 2))
print("Waktu/frame (ms):", round((end_time - start_time) / frame_id * 1000, 2))
