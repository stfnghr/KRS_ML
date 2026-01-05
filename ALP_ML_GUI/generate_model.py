import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("⏳ Sedang memproses data & melatih model... (Tunggu sebentar)")

# 1. Load Data
# Pastikan adult.csv ada di folder yang sama!
try:
    df = pd.read_csv("adult.csv", encoding="latin1")
except FileNotFoundError:
    print("❌ Error: File 'adult.csv' tidak ditemukan. Pastikan file dataset ada di folder ini.")
    exit()

# 2. Cleaning & Encoding (Sesuai Notebook)
df = df.replace('?', np.nan)
for col in ['workclass', 'occupation', 'native.country']:
    df[col] = df[col].fillna(df[col].mode()[0])

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
df['relationship_status'] = df['relationship'].apply(lambda x: 1 if x in ['Husband', 'Wife'] else 0)
df['race_grouped'] = df['race'].apply(lambda x: 1 if x == 'White' else 0)
df['is_married'] = df['marital.status'].apply(lambda x: 1 if x.startswith('Married') else 0)

# Drop kolom yang sudah di-engineer atau tidak dipakai
df = df.drop(['fnlwgt', 'relationship', 'race', 'marital.status'], axis=1)

# One-Hot Encoding
df = pd.get_dummies(df, columns=['workclass', 'education', 'occupation', 'native.country'], drop_first=True)

# 3. Split Data
X = df.drop('income', axis=1)
y = df['income']

# Simpan daftar nama kolom (PENTING untuk UI)
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scaling
scaler = StandardScaler()
numerical_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 
                      'sex', 'is_married', 'relationship_status', 'race_grouped']

# Pastikan kolom numerik ada di X_train sebelum scaling
valid_num_features = [col for col in numerical_features if col in X_train.columns]

X_train[valid_num_features] = scaler.fit_transform(X_train[valid_num_features])
joblib.dump(scaler, 'scaler.pkl') # Simpan Scaler

# 5. Training Model (Random Forest Tuned)
# Parameter sesuai hasil tuning notebook kamu
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
joblib.dump(rf, 'best_rf_model.pkl') # Simpan Model

print("✅ SUKSES! File model berhasil dibuat:")
print("   1. best_rf_model.pkl")
print("   2. scaler.pkl")
print("   3. model_columns.pkl")
print("Silakan refresh halaman Streamlit kamu sekarang.")