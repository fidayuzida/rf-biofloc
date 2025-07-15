# ================================================================================
# 0. IMPORT LIBRARY
# ================================================================================
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import kagglehub

# ================================================================================
# 1. BACA & GABUNGKAN DATASET
# ================================================================================
path = kagglehub.dataset_download("ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df_list = [pd.read_csv(file, low_memory=False) for file in csv_files]
df_combined = pd.concat(df_list, ignore_index=True)
print(f"Total data: {len(df_combined)} baris dari {len(csv_files)} file.")

# ================================================================================
# 2. STANDARISASI KOLOM & CLEANING
# ================================================================================
temp_cols = ['Temperature(C)', 'TEMPERATURE', 'Temperature (C)', 'temperature(C)']
turb_cols = ['Turbidity(NTU)', 'TURBIDITY', 'Turbidity (NTU)', 'turbidity (NTU)']
ph_cols = ['PH', 'pH']
date_cols = ['created_at', 'Date', 'DATE']

def ambil_nilai_pertama(row, kolom_list):
    for col in kolom_list:
        if col in row and pd.notnull(row[col]):
            return row[col]
    return np.nan

df_combined['Temperature'] = df_combined.apply(lambda row: ambil_nilai_pertama(row, temp_cols), axis=1)
df_combined['Turbidity'] = df_combined.apply(lambda row: ambil_nilai_pertama(row, turb_cols), axis=1)
df_combined['pH'] = df_combined.apply(lambda row: ambil_nilai_pertama(row, ph_cols), axis=1)

df_combined['datetime_clean'] = df_combined.apply(lambda row: ambil_nilai_pertama(row, date_cols), axis=1)
df_combined['datetime_clean'] = pd.to_datetime(df_combined['datetime_clean'], errors='coerce', dayfirst=True, utc=True)
df_combined['datetime_clean'] = df_combined['datetime_clean'].dt.tz_convert('Asia/Jakarta')
df_combined.dropna(subset=['datetime_clean'], inplace=True)

df_combined = df_combined[(df_combined['Temperature'] > 0) & (df_combined['Temperature'] < 50)]

# ================================================================================
# 3. FEATURE ENGINEERING & LABELING
# ================================================================================
df_combined['Jam'] = df_combined['datetime_clean'].dt.hour

def get_part_of_day(hour):
    if 5 <= hour < 12: return 'Pagi'
    elif 12 <= hour < 17: return 'Siang'
    elif 17 <= hour < 21: return 'Sore'
    else: return 'Malam'

df_combined['Bagian_Hari'] = df_combined['Jam'].apply(get_part_of_day)

features_to_use = ['Temperature', 'Turbidity', 'pH', 'Jam', 'Bagian_Hari']
df_processed = df_combined[features_to_use].copy()
df_processed.dropna(inplace=True)

def label_kualitas_air(row):
    skor = 0
    if 6.5 <= row['pH'] <= 8.5:
        skor += 0.4
    if 20 <= row['Temperature'] <= 30:
        skor += 0.4
    if row['Turbidity'] < 50:
        skor += 0.2
    return 'Baik' if skor >= 0.6 else 'Buruk'

df_processed['Kualitas'] = df_processed.apply(label_kualitas_air, axis=1)

# ================================================================================
# 4. ENCODING, SCALING, SPLIT
# ================================================================================
X = df_processed.drop('Kualitas', axis=1)
X = pd.get_dummies(X, columns=['Bagian_Hari'], drop_first=True)
y = df_processed['Kualitas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
numerical_cols = ['Temperature', 'Turbidity', 'pH', 'Jam']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ================================================================================
# 5. TRAIN MODEL
# ================================================================================
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ================================================================================
# 6. EVALUASI MODEL
# ================================================================================
y_pred = rf.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Confusion Matrix RF (WIB)")
plt.tight_layout()
plt.show()

# ================================================================================
# 7. FEATURE IMPORTANCE
# ================================================================================
importances = rf.feature_importances_
feature_names = X.columns
imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(imp_df['Feature'], imp_df['Importance'])
plt.title('Feature Importance RF (WIB)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ================================================================================
# 8. CROSS-VALIDATION & EXPORT
# ================================================================================
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Cross-Val Accuracy:", cv_scores)
print("Mean:", np.mean(cv_scores), "Std:", np.std(cv_scores))

joblib.dump(rf, "model_rf_wib.pkl")
joblib.dump(scaler, "scaler_rf_wib.pkl")
joblib.dump(list(X.columns), "feature_columns_rf_wib.pkl")
print("✅ Model + WIB berhasil dilatih dan disimpan.")
