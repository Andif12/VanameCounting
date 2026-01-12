import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ==============================
# CONFIG
# ==============================
RAW_DIR = Path("verifier_raw")          # hasil crop dari Roboflow
OUT_DIR = Path("dataset_verifier")      # output dataset verifier

TOTAL_SAMPLES = 2000                    # jumlah subset
TRAIN_RATIO = 0.8                       # 80% train, 20% val

INVALID_TARGET_RATIO = 0.5              # target rasio invalid
INVALID_SIZE = (32, 32)                 # ukuran rusak

random.seed(42)
np.random.seed(42)

# ==============================
# RESET OUTPUT FOLDER
# ==============================
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for split in ["train", "val"]:
    for cls in ["benur_valid", "benur_invalid"]:
        (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD SUBSET
# ==============================
all_images = list(RAW_DIR.glob("*.jpg"))
if len(all_images) == 0:
    raise RuntimeError("Folder verifier_raw kosong")

random.shuffle(all_images)
selected = all_images[:TOTAL_SAMPLES]

train_cut = int(TRAIN_RATIO * TOTAL_SAMPLES)

# ==============================
# MAKE INVALID IMAGE (AMAN)
# ==============================
def make_invalid(img):
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]

    # blur
    img = cv2.GaussianBlur(img, (15, 15), 0)

    # random crop aman
    if h > 20 and w > 20:
        x1 = random.randint(0, w // 4)
        y1 = random.randint(0, h // 4)
        x2 = random.randint(max(x1 + 10, w * 3 // 4), w)
        y2 = random.randint(max(y1 + 10, h * 3 // 4), h)
        cropped = img[y1:y2, x1:x2]
        if cropped.size > 0:
            img = cropped

    # resize ekstrem (selalu valid)
    try:
        img = cv2.resize(img, INVALID_SIZE)
    except:
        return None

    if img is None or img.size == 0:
        return None

    return img

# ==============================
# BUILD DATASET
# ==============================
valid_count = 0
invalid_count = 0

for i, img_path in enumerate(tqdm(selected, desc="Building verifier dataset")):
    split = "train" if i < train_cut else "val"

    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        continue

    # ===== VALID (ASLI) =====
    valid_dst = OUT_DIR / split / "benur_valid" / img_path.name
    cv2.imwrite(str(valid_dst), img)
    valid_count += 1

    # ===== INVALID (RUSAK BUATAN) =====
    if random.random() < INVALID_TARGET_RATIO:
        invalid_img = make_invalid(img)
        if invalid_img is not None:
            invalid_name = img_path.stem + "_invalid.jpg"
            invalid_dst = OUT_DIR / split / "benur_invalid" / invalid_name
            cv2.imwrite(str(invalid_dst), invalid_img)
            invalid_count += 1

# ==============================
# SUMMARY
# ==============================
print("\n===== DATASET VERIFIER SUMMARY =====")
print(f"Total raw images       : {len(all_images)}")
print(f"Total selected samples : {TOTAL_SAMPLES}")
print(f"Benur VALID            : {valid_count}")
print(f"Benur INVALID          : {invalid_count}")

print("\nStruktur akhir:")
print("dataset_verifier/")
print(" ├── train/")
print(" │    ├── benur_valid/")
print(" │    └── benur_invalid/")
print(" └── val/")
print("      ├── benur_valid/")
print("      └── benur_invalid/")
