from flask import Flask, request, jsonify
import joblib
import pandas as pd

# --- 1. Muat 3 file yang dibutuhkan untuk model baru ---
try:
    model = joblib.load('model_rf_wib.pkl')
    scaler = joblib.load('scaler_rf_wib.pkl')
    feature_columns = joblib.load('feature_columns_rf_wib.pkl')
except FileNotFoundError as e:
    model, scaler, feature_columns = None, None, None
    print(f"Error memuat file: {e}")

# --- 2. Inisialisasi Flask App ---
app = Flask(__name__)

# --- 3. Rute untuk Prediksi ---
@app.route('/', methods=['POST'])
def predict():
    if not all([model, scaler, feature_columns]):
        return jsonify({"error": "Model, Scaler, atau Kolom Fitur tidak ditemukan di server."}), 500

    if request.method != 'POST':
        return 'Metode tidak diizinkan', 405

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body harus dalam format JSON."}), 400

    try:
        # Ubah data mentah dari server menjadi DataFrame
        df_live = pd.DataFrame([data])
        
        # Konversi timestamp ke datetime dan ambil jam
        df_live['datetime_clean'] = pd.to_datetime(df_live['timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce').dt.tz_localize('Asia/Jakarta')
        df_live['Jam'] = df_live['datetime_clean'].dt.hour

        # Buat fitur bagian hari
        def get_part_of_day(hour):
            if 5 <= hour < 12: return 'Pagi'
            elif 12 <= hour < 17: return 'Siang'
            elif 17 <= hour < 21: return 'Sore'
            else: return 'Malam'
        df_live['Bagian_Hari'] = df_live['Jam'].apply(get_part_of_day)

        # Rename agar konsisten
        if 'temperature' in df_live.columns:
            df_live.rename(columns={'temperature': 'Temperature'}, inplace=True)

        # One-hot encoding dan reindex
        X_live = pd.get_dummies(df_live[['Temperature', 'Turbidity', 'pH', 'Jam', 'Bagian_Hari']], columns=['Bagian_Hari'])
        X_live = X_live.reindex(columns=feature_columns, fill_value=0)

        # Scaling
        numerical_cols = ['Temperature', 'Turbidity', 'pH', 'Jam']
        X_live[numerical_cols] = scaler.transform(X_live[numerical_cols])

        # Prediksi
        prediction = model.predict(X_live)[0]
        return jsonify({'label': str(prediction)})

    except Exception as e:
        return jsonify({"error": f"Gagal melakukan prediksi: {e}"}), 500

# --- 4. Jalankan Server Lokal ---
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8081)
