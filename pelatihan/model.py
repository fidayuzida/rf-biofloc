# PART 1: IMPORT & LOAD DATASET
import pandas as pd
import numpy as np
import glob
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import kagglehub

print("[INFO] Mengunduh dan membaca dataset Kaggle...")
path = kagglehub.dataset_download("ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets")
csv_files = glob.glob(os.path.join(path, "*.csv"))
csv_files.append("/content/synthetic_buruk_10000.csv")

list_df_kolam_processed = []

def get_first_valid(row, cols):
    for col in cols:
        if col in row.index and pd.notnull(row[col]):
            return row[col]
    return np.nan

for file in csv_files:
    df = pd.read_csv(file, low_memory=False)
    df['temperature'] = df.apply(lambda row: get_first_valid(row, ['Temperature(C)', 'TEMPERATURE', 'Temperature (C)', 'temperature(C)']), axis=1)
    df['turbidity'] = df.apply(lambda row: get_first_valid(row, ['Turbidity(NTU)', 'TURBIDITY', 'Turbidity (NTU)', 'turbidity (NTU)']), axis=1)
    df['ph'] = df.apply(lambda row: get_first_valid(row, ['PH', 'pH']), axis=1)
    df['datetime_clean'] = df.apply(lambda row: get_first_valid(row, ['created_at', 'Date', 'DATE']), axis=1)
    df['datetime_clean'] = df['datetime_clean'].astype(str).str.replace(' CET', '', regex=False)
    df['datetime_clean'] = pd.to_datetime(df['datetime_clean'], errors='coerce')
    df.dropna(subset=['datetime_clean'], inplace=True)
    if df['datetime_clean'].dt.tz is None:
        df['datetime_clean'] = df['datetime_clean'].dt.tz_localize('UTC')
    df['datetime_clean'] = df['datetime_clean'].dt.tz_convert('Asia/Jakarta')
    df.set_index('datetime_clean', inplace=True)
    df.sort_index(inplace=True)
    df_resampled = df[['temperature', 'ph', 'turbidity']].resample('3T').mean().interpolate(method='time').dropna()

    def label_kualitas_air(row):
        skor = 0
        if 6.5 <= row['ph'] <= 8.5:
            skor += 0.4
        if 20 <= row['temperature'] <= 30:
            skor += 0.4
        if row['turbidity'] < 50:
            skor += 0.2
        if skor >= 0.9:
            return "Baik"
        elif skor >= 0.6:
            return "Agak Baik"
        elif skor >= 0.3:
            return "Agak Buruk"
        else:
            return "Buruk"

    df_resampled['kualitas'] = df_resampled.apply(label_kualitas_air, axis=1)
    df_resampled['kualitas_future'] = df_resampled['kualitas'].shift(-2)

    for lag in range(1, 6):
        df_resampled[f'temp_lag_{lag}'] = df_resampled['temperature'].shift(lag)
        df_resampled[f'ph_lag_{lag}'] = df_resampled['ph'].shift(lag)
        df_resampled[f'turb_lag_{lag}'] = df_resampled['turbidity'].shift(lag)

    for win in [2, 3, 5]:
        df_resampled[f'temp_roll_mean_{win}'] = df_resampled['temperature'].rolling(win).mean()
        df_resampled[f'ph_roll_std_{win}'] = df_resampled['ph'].rolling(win).std()
        df_resampled[f'turb_roll_mean_{win}'] = df_resampled['turbidity'].rolling(win).mean()

    df_resampled.dropna(inplace=True)
    list_df_kolam_processed.append(df_resampled)

df_model = pd.concat(list_df_kolam_processed, ignore_index=True)
print(f"[INFO] Total data setelah preprocessing: {len(df_model)}")

# PART 2: MODELING & EVALUASI

X = df_model.drop(columns=['kualitas', 'kualitas_future', 'temperature', 'ph', 'turbidity'])
y = df_model['kualitas_future']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=false, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[INFO] Menerapkan SMOTE untuk penyeimbangan kelas...")
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train_scaled, y_train)

print("[INFO] Melatih model Random Forest dengan hyperparameter terbaik (manual)...")

best_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train_resampled, y_train_resampled)
print("[INFO] Training selesai.")

# Cross-validation manual
print("[INFO] Melakukan 5-Fold Cross-Validation pada model terbaik...")
cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5, scoring='f1_weighted')
mean_score = np.mean(cv_scores)
std_score = np.std(cv_scores)

for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: F1-Weighted Score = {score:.4f}")
print(f"Rata-rata F1-Weighted Score: {mean_score:.4f}")
print(f"Standar Deviasi: {std_score:.4f}")

# PART 3: EVALUASI MODEL & VISUALISASI UTAMA

# Evaluasi akhir
print("[INFO] Evaluasi model pada data uji...")
y_pred = best_model.predict(X_test_scaled)
print("--- Classification Report ---\n", classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test)
plt.title("Confusion Matrix (4 Kelas - Prediksi 6 Menit ke Depan)")
plt.tight_layout()
plt.savefig("confusion_matrix_4class.png")
plt.show()
plt.close()

# Feature importance
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(y=feat_imp_df['Feature'][:15], x=feat_imp_df['Importance'][:15])
plt.title("Top 15 Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance_4class.png")
plt.close()

# 1. Distribusi Setelah Labelling
label_counts = df_model['kualitas_future'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(6, 4))
ax = label_counts.plot(kind='bar', color='skyblue')
plt.title("Distribusi Label Setelah Labelling")
plt.ylabel("Jumlah Sampel")
plt.xlabel("Label Kualitas Air (6 Menit ke Depan)")
plt.xticks(rotation=0)

for i, count in enumerate(label_counts):
    ax.text(i, count + 1000, f"{count:,}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("label_distribution_after_labelling_4class.png")
plt.close()


#2. Distribusi Sebelum SMOTE
pre_smote_counts = pd.Series(y_train).value_counts().sort_values(ascending=False)

plt.figure(figsize=(6, 4))
ax = pre_smote_counts.plot(kind='bar', color='skyblue')
plt.title("Distribusi Label Sebelum SMOTE")
plt.ylabel("Jumlah Sampel")
plt.xlabel("Label")
plt.xticks(rotation=0)

for i, count in enumerate(pre_smote_counts):
    #ax.text(i, count + 1000, f"{count:,}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("label_distribution_before_smote_4class.png")
plt.close()


# 3. Distribusi Setelah SMOTE
post_smote_counts = pd.Series(y_train_resampled).value_counts().sort_values(ascending=False)

plt.figure(figsize=(6, 4))
ax = post_smote_counts.plot(kind='bar', color='skyblue')
plt.title("Distribusi Label Setelah SMOTE")
plt.ylabel("Jumlah Sampel")
plt.xlabel("Label")
plt.xticks(rotation=0)

for i, count in enumerate(post_smote_counts):
    #ax.text(i, count + 1000, f"{count:,}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("label_distribution_after_smote_4class.png")
plt.close()


# Simpan model & laporan
joblib.dump(best_model, "model_rf_4class.pkl")
joblib.dump(scaler, "scaler_rf_4class.pkl")
joblib.dump(X.columns.to_list(), "features_rf_4class.pkl")

with open("final_evaluation_summary_4class.txt", "w") as f:
    f.write("=== EVALUASI MODEL RANDOM FOREST (4 KELAS) ===\n\n")
    f.write("=== CROSS-VALIDATION (5-FOLD) ===\n")
    for i, score in enumerate(cv_scores):
        f.write(f"Fold {i+1}: F1-Weighted = {score:.4f}\n")
    f.write(f"\nRata-rata F1-Weighted Score: {mean_score:.4f}\n")
    f.write(f"Standar Deviasi: {std_score:.4f}\n")
    f.write("\nBest Parameters (manual setting):\n")
    f.write("{'n_estimators': 100, 'max_depth': 20, 'min_samples_leaf': 5, 'min_samples_split': 10}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))
    f.write("\n\nTop 10 Feature Importance:\n")
    for _, row in feat_imp_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")


# PART 4: VISUALISASI TAMBAHAN UNTUK DOKUMENTASI

# Missing value visualisasi
plt.figure()
msno.matrix(df_model[['temperature', 'ph', 'turbidity']])
plt.title("Visualisasi Missing Value (Matrix)")
plt.savefig("missing_value_matrix_4class.png")
plt.close()

plt.figure()
msno.bar(df_model[['temperature', 'ph', 'turbidity']])
plt.title("Distribusi Missing Value per Parameter")
plt.tight_layout()
plt.savefig("missing_value_bar_rawdata.png")
plt.close()

# Skor kualitas distribusi
skor_mapping = {'Baik': 4, 'Agak Baik': 3, 'Agak Buruk': 2, 'Buruk': 1}
df_model['skor_kualitas'] = df_model['kualitas_future'].map(skor_mapping)

plt.figure()
sns.histplot(df_model['skor_kualitas'].dropna(), bins=4, kde=False)
plt.title("Distribusi Skor Kualitas Air")
plt.xlabel("Skor Kualitas (1=Buruk, 4=Baik)")
plt.xticks([1, 2, 3, 4], ['Buruk', 'Agak Buruk', 'Agak Baik', 'Baik'])
plt.tight_layout()
plt.savefig("distribusi_skor_kualitas_air.png")
plt.close()

# Violin plot sensor mentah
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_model[['temperature', 'ph', 'turbidity']])
plt.title("Distribusi Sensor Mentah (Violin Plot)")
plt.ylabel("Nilai Sensor")
plt.tight_layout()
plt.savefig("violin_sensor_mentah_4class.png")
plt.close()

# KDE plot dengan rentang ideal
plt.figure()
sns.kdeplot(df_model['temperature'], fill=True)
plt.axvspan(20, 30, color='green', alpha=0.2, label='Rentang Ideal')
plt.title("Distribusi Suhu dengan Rentang Ideal")
plt.xlabel("Suhu (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("kde_temperature_4class.png")
plt.close()

plt.figure()
sns.kdeplot(df_model['ph'], fill=True, color='orange')
plt.axvspan(6.5, 8.5, color='green', alpha=0.2, label='Rentang Ideal')
plt.title("Distribusi pH dengan Rentang Ideal")
plt.xlabel("pH")
plt.legend()
plt.tight_layout()
plt.savefig("kde_ph_4class.png")
plt.close()

plt.figure()
sns.kdeplot(df_model['turbidity'], fill=True, color='purple')
plt.axvspan(0, 50, color='green', alpha=0.2, label='Rentang Ideal')
plt.title("Distribusi Kekeruhan dengan Rentang Ideal")
plt.xlabel("Turbidity (NTU)")
plt.legend()
plt.tight_layout()
plt.savefig("kde_turbidity_4class.png")
plt.close()

# Before-after resampling suhu
df_raw_first = pd.read_csv(csv_files[0])
df_raw_first['datetime_clean'] = df_raw_first.apply(lambda row: get_first_valid(row, ['created_at', 'Date', 'DATE']), axis=1)
df_raw_first['datetime_clean'] = pd.to_datetime(df_raw_first['datetime_clean'], errors='coerce')
df_raw_first.set_index('datetime_clean', inplace=True)
df_raw_first.sort_index(inplace=True)

plt.figure()
df_raw_first['Temperature(C)'].plot()
plt.title("Suhu Sebelum Resampling (Data Mentah Kolam 1)")
plt.xlabel("Waktu")
plt.ylabel("Suhu (°C)")
plt.tight_layout()
plt.savefig("temperature_before_resampling_4class.png")
plt.close()

plt.figure()
df_model['temperature'].plot()
plt.title("Suhu Setelah Resampling (Gabungan Kolam)")
plt.xlabel("Waktu")
plt.ylabel("Suhu (°C)")
plt.tight_layout()
plt.savefig("temperature_after_resampling_4class.png")
plt.close()

# Kompres semua hasil ke zip
import zipfile
output_files = [
    "model_rf_4class.pkl", "scaler_rf_4class.pkl", "features_rf_4class.pkl",
    "final_evaluation_summary_4class.txt", "confusion_matrix_4class.png",
    "feature_importance_4class.png", "label_distribution_after_labelling_4class.png",
    "label_distribution_before_smote_4class.png", "label_distribution_after_smote_4class.png",
    "distribusi_skor_kualitas_air.png", "violin_sensor_mentah_4class.png",
    "kde_temperature_4class.png", "kde_ph_4class.png", "kde_turbidity_4class.png",
    "missing_value_bar_rawdata.png", "missing_value_matrix_4class.png",
    "temperature_before_resampling_4class.png", "temperature_after_resampling_4class.png"
]

with zipfile.ZipFile("hasil_prediksi_rf_4kelas.zip", "w") as zipf:
    for file in output_files:
        if os.path.exists(file):
            zipf.write(file)

print("[ZIP] Semua file hasil telah dikompres menjadi 'hasil_prediksi_rf_4kelas.zip'")
