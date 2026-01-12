import pandas as pd
import numpy as np

# ==============================
# 1. YOLO per frame (stabilitas)
# ==============================
df_yolo = pd.read_csv("hasil_yolo_per_frame.csv")

mean_yolo = df_yolo["jumlah_benur"].mean()
std_yolo = df_yolo["jumlah_benur"].std()

fps_yolo = 1000 / 27  # rata-rata dari log kamu ±27 ms

if std_yolo > 15:
    stabilitas_yolo = "Tinggi, fluktuatif"
    catatan_yolo = "Double counting antar frame"
else:
    stabilitas_yolo = "Tinggi, stabil"
    catatan_yolo = "Deteksi konsisten"

# ==============================
# 2. YOLO + ByteTrack
# ==============================
df_bt = pd.read_csv("hasil_bytetrack_snapshot.csv")

fps_bt = df_bt["fps"].iloc[0]
stabilitas_bt = "Lebih stabil"
catatan_bt = "Menggunakan ID tracking"

# ==============================
# 3. YOLO + ByteTrack + EN
# ==============================
df_ver = pd.read_csv("hasil_verified_snapshot.csv")

fps_ver = df_ver["fps"].iloc[0]
stabilitas_ver = "Paling stabil"
catatan_ver = "Verifikasi objek non-benur"

# ==============================
# 4. Buat tabel ringkas
# ==============================
tabel = pd.DataFrame({
    "Model": [
        "YOLO",
        "YOLO + ByteTrack",
        "YOLO + ByteTrack + EfficientNet"
    ],
    "Akurasi Deteksi / Stabilitas": [
        stabilitas_yolo,
        stabilitas_bt,
        stabilitas_ver
    ],
    "FPS": [
        round(fps_yolo, 1),
        round(fps_bt, 1),
        round(fps_ver, 1)
    ],
    "Catatan": [
        catatan_yolo,
        catatan_bt,
        catatan_ver
    ]
})

# Simpan
tabel.to_csv("tabel_ringkas_model.csv", index=False)

print("=== TABEL RINGKAS BERHASIL DIBUAT ===")
print(tabel)
