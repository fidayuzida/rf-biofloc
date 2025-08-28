# 🌊 Bioflok Water Quality Prediction

## 📌 Overview
Proyek ini merupakan Tugas Akhir dengan judul **“Penerapan Algoritma Random Forest untuk Prediksi Kualitas Air Kolam Bioflok dalam Budidaya Lele”**.  
Sistem ini menggabungkan **IoT sensor, Machine Learning, dan Web Application** untuk memprediksi kualitas air kolam bioflok **6 menit ke depan** secara real-time.

## 🎯 Objectives
- Memantau parameter kualitas air (suhu, pH, kekeruhan) secara otomatis.
- Mengklasifikasikan kondisi air ke dalam 4 kategori: `Baik`, `Agak Baik`, `Agak Buruk`, `Buruk`.
- Menyediakan dashboard web interaktif dengan log historis, grafik, filter tanggal, dan ekspor CSV.

## ⚙️ System Architecture
### 1. IoT & Hardware
- **ESP32** sebagai mikrokontroler.
- Sensor: **DS18B20 (suhu)**, **pH-4502C**, **SEN0189 (kekeruhan)**.
- Data dikirim setiap 3 menit ke **Firebase Realtime Database**.

### 2. Machine Learning
- Algoritma: **Random Forest Classifier**.
- Preprocessing: standarisasi, resampling, interpolasi missing value.
- Feature engineering: lag features, rolling mean/std.
- Handling imbalance: **SMOTE**.
- Hyperparameter tuning: **Randomized Search**.
- **Hasil:** Akurasi 98.05%, rata-rata F1-score 0.9478.

### 3. Web Application
- **Backend**: Flask API (Python, scikit-learn) + Node.js (Express) untuk komunikasi data.
- **Frontend**: HTML + Bootstrap (Firebase Hosting).
- **Deployment**: Azure VM + Ngrok tunneling.
- Fitur: dashboard real-time, log historis, filter tanggal, ekspor CSV.

## 🚀 Features
- 📡 Real-time monitoring suhu, pH, kekeruhan.
- 🤖 Prediksi kualitas air 6 menit ke depan.
- 📊 Visualisasi grafik interaktif & tabel log.
- ⏱️ Filter data berdasarkan rentang tanggal.
- 📂 Ekspor data ke CSV.
- ☁️ Full-stack integrasi IoT + ML + Web.

## 🛠️ Tech Stack
- **IoT**: ESP32, Arduino IDE
- **Machine Learning**: Python, scikit-learn, Pandas, SMOTE
- **Web**: Flask, Node.js, Express, Bootstrap, Firebase Hosting
- **Cloud**: Firebase Realtime Database, Azure VM, Ngrok
- **Hardware**: Sensor pH-4502C, DS18B20, SEN0189

## 🔗 Links
- 🌐 Website Monitoring: http://rf-bioflok.web.app/

