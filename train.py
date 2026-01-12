from ultralytics import YOLO

# =====================
# CONFIG
# =====================
MODEL_PATH = "model/yolo11n.pt"
DATA_YAML = "D:\\KULIAH FIRA\\about skripsi fiks\\kode\\multi_model\\Pl10-5\\data.yaml"

# =====================
# LOAD MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# TRAIN (CPU MODE)
# =====================
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",      # 🔴 WAJIB CPU
    name="benur_yolo"
)
