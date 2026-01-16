from ultralytics import YOLO

MODEL_PATH = "model/yolo11n.pt"
DATA_YAML = "D:\\KULIAH FIRA\\about skripsi fiks\\kode\\multi_model\\Pl10-5\\data.yaml"

model = YOLO(MODEL_PATH)

model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",   
    name="benur_yolo"
)
