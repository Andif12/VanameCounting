# VenameCounting
**DETEKSI DAN PENGHITUNGAN BENUR UDANG VANAME MENGGUNAKAN PENDEKATAN TERINTEGRASI MULTI-MODEL**

## 📌 Deskripsi
**VenameCounting** merupakan *source code* sistem **deteksi dan penghitungan benur udang vaname (*Litopenaeus vannamei*)** yang dikembangkan sebagai bagian dari **penelitian skripsi**.  
Sistem ini dirancang untuk bekerja secara **otomatis pada kondisi nyata hatchery** dengan memanfaatkan pendekatan **multi-model computer vision** yang terintegrasi melalui arsitektur layanan (*service-based architecture*).

Sistem mengombinasikan **YOLO** untuk deteksi objek, **ByteTrack** untuk pelacakan antar frame, serta **EfficientNet** (opsional) sebagai *verifier* guna meningkatkan keandalan hasil penghitungan.

---

## 🎯 Tujuan Pengembangan
- Mengimplementasikan sistem otomatis penghitungan benur udang vaname berbasis pendekatan multi-model
- Mengintegrasikan layanan frontend, backend, dan AI secara terstruktur
- Menghasilkan output jumlah benur yang akurat dan mudah dipahami pengguna
- Mendukung optimalisasi proses produksi hatchery melalui penghitungan yang lebih efisien dan presisi

---

## 🧠 Alur Implementasi Sistem
Sistem dibangun menggunakan arsitektur **client–server berbasis REST API** dengan alur sebagai berikut:

### 1️⃣ Unggah Video (Frontend)
Pengguna mengunggah video benur melalui **aplikasi Flutter** sebagai antarmuka utama sistem.

### 2️⃣ Backend Laravel
Video dikirim ke **backend Laravel** melalui RESTful API, kemudian diteruskan ke layanan AI untuk diproses.

### 3️⃣ Layanan AI (FastAPI – Python)
Backend Laravel memanggil **layanan FastAPI** yang menangani proses deteksi, pelacakan, dan penghitungan benur.

### 4️⃣ Deteksi Objek (YOLO)
Model **YOLO** digunakan untuk mendeteksi benur pada setiap frame video secara real-time atau batch.

### 5️⃣ Pelacakan Objek (ByteTrack)
**ByteTrack** melacak benur antar frame menggunakan ID unik, sehingga:
- Menghindari duplikasi hitungan
- Menjaga konsistensi identitas objek

### 6️⃣ Verifikasi Deteksi (Opsional – EfficientNet)
Model **EfficientNet** digunakan sebagai *verifier* untuk memvalidasi hasil deteksi YOLO, khususnya pada kondisi:
- Kepadatan benur tinggi
- Overlapping objek
- Noise visual

### 7️⃣ Penghitungan Benur
Jumlah benur dihitung berdasarkan lintasan objek (*tracking-based counting*), bukan sekadar jumlah deteksi per frame.

### 8️⃣ Output JSON
Hasil pengolahan dikembalikan ke backend Laravel dalam format **JSON** yang terstruktur.

### 9️⃣ Visualisasi Hasil
Backend mengirim hasil akhir ke **aplikasi Flutter**, dan jumlah benur ditampilkan kepada pengguna dengan tampilan yang mudah dipahami.

---

## 🧩 Arsitektur Sistem
```text
Flutter App
    ↓
Laravel Backend (REST API)
    ↓
FastAPI (AI Service)
    ├── YOLO (Object Detection)
    ├── ByteTrack (Multi-Object Tracking)
    └── EfficientNet (Detection Verifier - Optional)
    ↓
JSON Output
    ↓
Laravel Backend
    ↓
Flutter App (Visualization)
