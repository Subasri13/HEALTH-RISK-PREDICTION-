"""
AI Health Risk Prediction - Model Training Script
Run this once to generate the trained model files.

Usage: python train_model.py
Requirements: pip install scikit-learn pandas numpy joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset/symptom_disease_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../backend/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Dataset ───────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")

# ── Features & Labels ─────────────────────────────────────────────────────────
SYMPTOM_COLS = [
    "fever", "cough", "headache", "fatigue", "nausea",
    "chest_pain", "shortness_of_breath", "sore_throat", "body_ache",
    "diarrhea", "vomiting", "runny_nose", "dizziness", "rash", "loss_of_appetite"
]

X = df[SYMPTOM_COLS]
y_disease = df["disease"]
y_risk    = df["risk_level"]

# ── Encode Labels ─────────────────────────────────────────────────────────────
disease_encoder = LabelEncoder()
risk_encoder    = LabelEncoder()

y_disease_enc = disease_encoder.fit_transform(y_disease)
y_risk_enc    = risk_encoder.fit_transform(y_risk)

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, yd_train, yd_test, yr_train, yr_test = train_test_split(
    X, y_disease_enc, y_risk_enc, test_size=0.2, random_state=42
)

# ── Disease Prediction Model ───────────────────────────────────────────────────
print("\n🤖 Training Disease Prediction Model (Random Forest)...")
disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
disease_model.fit(X_train, yd_train)
yd_pred = disease_model.predict(X_test)
print(f"   Disease Model Accuracy: {accuracy_score(yd_test, yd_pred):.2%}")
print(classification_report(yd_test, yd_pred))

# ── Risk Level Model ───────────────────────────────────────────────────────────
print("\n⚠️  Training Risk Level Model (Random Forest)...")
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, yr_train)
yr_pred = risk_model.predict(X_test)
print(f"   Risk Model Accuracy: {accuracy_score(yr_test, yr_pred):.2%}")
print(classification_report(yr_test, yr_pred))

# ── Save Models & Encoders ────────────────────────────────────────────────────
print("\n💾 Saving models...")
joblib.dump(disease_model,    os.path.join(MODEL_DIR, "disease_model.pkl"))
joblib.dump(risk_model,       os.path.join(MODEL_DIR, "risk_model.pkl"))
joblib.dump(disease_encoder,  os.path.join(MODEL_DIR, "disease_encoder.pkl"))
joblib.dump(risk_encoder,     os.path.join(MODEL_DIR, "risk_encoder.pkl"))

# Save symptom columns list for the API
with open(os.path.join(MODEL_DIR, "symptoms.json"), "w") as f:
    json.dump(SYMPTOM_COLS, f)

print("✅ All models saved to backend/models/")
print("\n🚀 You can now run the Flask server: cd backend && python app.py")
