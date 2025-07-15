import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Set seed
np.random.seed(42)

# 500 data realistis (dalam rentang optimal budidaya lele)
temp1 = np.random.uniform(20, 30, 500)
ph1 = np.random.uniform(6.5, 8.5, 500)
turb1 = np.random.uniform(0, 40, 500)

# 500 data ekstrem (di luar rentang optimal)
temp2 = np.random.uniform(-10, 50, 500)
ph2 = np.random.uniform(0, 14, 500)
turb2 = np.random.uniform(50, 3000, 500)

# Gabungkan jadi 1000 data
temperature = np.concatenate([temp1, temp2])
ph = np.concatenate([ph1, ph2])
turbidity = np.concatenate([turb1, turb2])

# Fungsi labeling: mayoritas parameter baik → 'Baik', sisanya 'Buruk'
def label_kualitas_air(temperature, ph, turbidity):
    skor_baik = 0
    if 6.5 <= ph <= 8.5:
        skor_baik += 1
    if 20 <= temperature <= 30:
        skor_baik += 1
    if turbidity < 500:
        skor_baik += 1
    return 'Baik' if skor_baik >= 2 else 'Buruk'

# Terapkan fungsi labeling
labels = [label_kualitas_air(temperature[i], ph[i], turbidity[i]) for i in range(1000)]

# Buat DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'ph': ph,
    'turbidity': turbidity,
    'label': labels
})

# Fitur dan target
X = df[['temperature', 'ph', 'turbidity']]
y = df['label']

# Split data (train: 80%, test: 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, 'random_forest_model.pkl')
print("\nModel disimpan sebagai 'random_forest_model.pkl'")
