from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "model/best.pt"
VIDEO_PATH = "Data/IMG_8987.MOV"

WINDOW_NAME = "YOLO Detection"

WINDOW_SIZE = 900      
IMG_SIZE = 1280       # ukuran input YOLO
CONF_THRESHOLD = 0.4

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Gagal membuka video")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_SIZE, WINDOW_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # YOLO inference
    results = model(
        frame,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        rect=True
    )

    annotated = frame.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]

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

    scale = min(WINDOW_SIZE / orig_w, WINDOW_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(annotated, (new_w, new_h))

    canvas = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    x_offset = (WINDOW_SIZE - new_w) // 2
    y_offset = (WINDOW_SIZE - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    cv2.imshow(WINDOW_NAME, canvas)

    # Exit dengan Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit jika window ditutup
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("Program selesai dengan aman.")