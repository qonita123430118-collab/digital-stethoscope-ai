# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Loading dataset...")

# Load kedua file
df_normal   = pd.read_csv('ptbdb_normal.csv',   header=None)
df_abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)

print(f"Normal samples   : {len(df_normal)}")
print(f"Abnormal samples : {len(df_abnormal)}")

# Gabungkan
df = pd.concat([df_normal, df_abnormal], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Pisahkan fitur dan label
X = df.iloc[:, :187].values   # 187 kolom sinyal
y = df.iloc[:, 187].values.astype(int)    # Kolom label (1=normal, 0=abnormal)

# Tambahkan verifikasi label setelah loading:
print(f"\nNilai unik label: {np.unique(y)}")
print(f"Distribusi label:\n{pd.Series(y).value_counts()}")

print(f"\nTotal samples : {len(X)}")
print(f"Normal        : {(y == 1).sum()}")
print(f"Abnormal      : {(y == 0).sum()}")

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,      # Tambahan: cegah overfitting
    class_weight='balanced', # Tambahan: handle imbalanced data
    random_state=42,
    n_jobs=-1       # Gunakan semua CPU core
)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("\n=== Hasil Evaluasi ===")
print(classification_report(y_test, y_pred,
      target_names=['Abnormal', 'Normal']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Simpan model dan scaler (keduanya wajib disimpan!)
joblib.dump(model,  'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel dan scaler berhasil disimpan!")