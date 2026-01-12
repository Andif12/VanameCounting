from ultralytics import YOLO
import cv2
import time
import pandas as pd

VIDEO_PATH = "Data/IMG_8988.MOV"
MODEL_PATH = "model/best.pt"
CONF = 0.25
MAX_FRAMES = 500   # samakan dengan YOLO per frame

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

unique_ids = set()
frame_count = 0

start_time = time.time()

while cap.isOpened() and frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        source=frame,
        conf=CONF,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        unique_ids.update(ids)

    frame_count += 1

cap.release()
end_time = time.time()

total_time = end_time - start_time
fps = frame_count / total_time

df = pd.DataFrame([{
    "total_benur": len(unique_ids),
    "fps": round(fps, 2),
    "total_frame": frame_count
}])

df.to_csv("hasil_bytetrack_snapshot.csv", index=False)

print("=== ByteTrack Selesai ===")
print(df)
