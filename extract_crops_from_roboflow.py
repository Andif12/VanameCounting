import cv2
from pathlib import Path
import os

# CONFIG (SUDAH SESUAI DATASET)
BASE = Path("Pl10-5")
SPLIT = "train"          # pakai data train saja
OUT_DIR = "verifier_raw"
IMG_EXT = [".jpg", ".jpeg", ".png"]

img_dir = BASE / SPLIT / "images"
label_dir = BASE / SPLIT / "labels"

if not img_dir.exists() or not label_dir.exists():
    raise RuntimeError("❌ Folder images / labels tidak ditemukan")

os.makedirs(OUT_DIR, exist_ok=True)

count = 0

print(f"✅ Images : {img_dir}")
print(f"✅ Labels : {label_dir}")

# EKSTRAK CROP
for img_path in img_dir.iterdir():
    if img_path.suffix.lower() not in IMG_EXT:
        continue

    label_path = label_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(label_path) as f:
        for i, line in enumerate(f):
            cls, x, y, bw, bh = map(float, line.strip().split())

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue

            out_name = f"{img_path.stem}_{i}.jpg"
            cv2.imwrite(str(Path(OUT_DIR) / out_name), crop)
            count += 1

print(f"\n🎯 Total crop benur diekstrak: {count}")
