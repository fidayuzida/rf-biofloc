# Impor library yang dibutuhkan
from flask import Flask, request, jsonify
import os
import joblib
import requests
import pandas as pd
import numpy as np
from flask_cors import CORS
from dotenv import load_dotenv

# --- Konfigurasi Awal ---
dotenv_path = os.path.join(os.path.dirname(__file__), '../main-app/.env')
load_dotenv(dotenv_path=dotenv_path)

FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")

app = Flask(__name__)
CORS(app) 

# === Muat Model dan Aset yang Sudah Dilatih ===
try:
    model = joblib.load('model_rf_4class.pkl')
    scaler = joblib.load('scaler_rf_4class.pkl')
    # Kita tidak perlu lagi memanipulasi feature_columns di sini
    feature_columns = joblib.load('features_rf_4class.pkl')
except FileNotFoundError as e:
    print(f"KRITIS: File model/scaler/features tidak ditemukan. Aplikasi tidak bisa berjalan. Error: {e}")
    exit()

# --- Fungsi Helper untuk Transformasi Data ---

def prepare_data_for_prediction(data_history):
    if len(data_history) < 6:
        print(f"LOG: Data tidak cukup. Butuh minimal 6, hanya ada {len(data_history)}.")
        return None

    df = pd.DataFrame(data_history)
    df = df.rename(columns={
        "temperature": "temp", 
        "pH": "ph", 
        "turbidity": "turb",
        "timestamp": "timestamp_str"
    })

    df['timestamp'] = pd.to_datetime(df['timestamp_str'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.set_index('timestamp').sort_index()
    
    numeric_cols = ['temp', 'ph', 'turb']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    # === Feature Engineering ===
    for i in range(1, 6):
        df[f'temp_lag_{i}'] = df['temp'].shift(i)
        df[f'ph_lag_{i}'] = df['ph'].shift(i)
        df[f'turb_lag_{i}'] = df['turb'].shift(i)
    
    for window in [2, 3, 5]:
        df[f'temp_roll_mean_{window}'] = df['temp'].shift(1).rolling(window=window).mean()
        df[f'turb_roll_mean_{window}'] = df['turb'].shift(1).rolling(window=window).mean()
        df[f'ph_roll_std_{window}'] = df['ph'].shift(1).rolling(window=window).std()

    df.fillna(0, inplace=True)
    df = df.dropna()
    
    if df.empty:
        print("LOG: Setelah feature engineering, tidak ada baris data yang valid.")
        return None
        
    # Ambil baris terakhir yang valid
    latest_features_df = df.iloc[-1:].copy()
    
    return latest_features_df

# --- Endpoints API ---

@app.route('/predict', methods=['GET'])
def predict():
    try:
        db_url = FIREBASE_DATABASE_URL.rstrip('/')
        firebase_url = f"{db_url}/sensor_readings.json?orderBy=\"$key\"&limitToLast=10"
        
        print(f"LOG: Mengambil data dari {firebase_url}")
        r = requests.get(firebase_url, timeout=10)
        r.raise_for_status()
        readings = r.json()

        if not isinstance(readings, dict) or not readings:
            return jsonify({'error': 'Tidak ada data sensor yang ditemukan di Firebase.'}), 404

        data_history = [value for key, value in sorted(readings.items())]
        df_to_predict = prepare_data_for_prediction(data_history)

        if df_to_predict is None:
            return jsonify({
                'error': 'Tidak cukup data historis untuk melakukan prediksi.',
                'message': f'Dibutuhkan minimal 6 data point, saat ini hanya ada {len(data_history)} yang valid.'
            }), 400
        
        # 1. Buat DataFrame sementara dengan semua kolom yang dibutuhkan dari file pkl
        final_df = pd.DataFrame(columns=feature_columns)
        
        # 2. Salin data dari df_to_predict ke final_df
        # Ini memastikan semua fitur input yang kita buat sudah ada
        for col in df_to_predict.columns:
            if col in final_df.columns:
                final_df[col] = df_to_predict[col]
        
        # 3. Isi kolom yang hilang (termasuk 'skor') dengan nilai 0
        final_df.fillna(0, inplace=True)

        # Sekarang `final_df` memiliki semua kolom yang diharapkan scaler, dengan urutan yang benar.
        df_scaled = scaler.transform(final_df)
        
        # Model.predict() tidak peduli dengan nama kolom, hanya urutan array numpy
        prediction_result = model.predict(df_scaled)[0]
        prediction_proba = model.predict_proba(df_scaled)[0]
        
        classes = model.classes_
        probabilities = {str(classes[i]): round(float(prediction_proba[i]), 4) for i in range(len(classes))}

        print(f"LOG: Prediksi berhasil -> {prediction_result}")
        return jsonify({
            'label': str(prediction_result),
            'probabilities': probabilities
        })

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Gagal menghubungi Firebase: {e}")
        return jsonify({'error': f"Gagal menghubungi Firebase: {e}"}), 500
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan tak terduga di /predict: {e}")
        return jsonify({'error': f"Terjadi kesalahan tak terduga: {e}"}), 500

# --- Menjalankan Aplikasi ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True)
