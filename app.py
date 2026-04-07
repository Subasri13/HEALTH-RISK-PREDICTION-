"""
AI Health Risk Prediction - Flask Backend API
==========================================
Run: python app.py
API runs on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DB_PATH    = os.path.join(BASE_DIR, "health_history.db")

# ── Load Models ───────────────────────────────────────────────────────────────
print("📦 Loading ML models...")
try:
    disease_model   = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
    risk_model      = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
    disease_encoder = joblib.load(os.path.join(MODEL_DIR, "disease_encoder.pkl"))
    risk_encoder    = joblib.load(os.path.join(MODEL_DIR, "risk_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "symptoms.json")) as f:
        SYMPTOM_COLS = json.load(f)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("   Please run: python model/train_model.py first")
    disease_model = risk_model = disease_encoder = risk_encoder = None
    SYMPTOM_COLS = []

# ── Recommendations Database ──────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Common Cold":    ["Rest at home", "Drink warm fluids", "Take OTC cold medicine", "Use steam inhalation"],
    "Influenza":      ["Bed rest for 5-7 days", "Stay hydrated", "Take antipyretics for fever", "Consult doctor if symptoms worsen"],
    "COVID-19":       ["Isolate immediately", "Monitor oxygen levels", "Consult a doctor", "Get tested", "Stay hydrated"],
    "Migraine":       ["Rest in a dark quiet room", "Apply cold compress", "Take prescribed migraine medicine", "Avoid screen time"],
    "Gastroenteritis":["Stay hydrated with ORS", "Eat bland food (BRAT diet)", "Rest", "Seek help if vomiting persists"],
    "Pneumonia":      ["Seek immediate medical attention", "Take prescribed antibiotics", "Complete bed rest", "Monitor breathing"],
    "Allergy":        ["Identify and avoid allergens", "Take antihistamines", "Consult allergist", "Carry emergency medication"],
    "Sinusitis":      ["Nasal saline rinse", "Steam inhalation", "Pain relievers", "Consult ENT if chronic"],
    "Typhoid":        ["Immediate medical consultation", "Take prescribed antibiotics", "Drink boiled water only", "Bed rest"],
    "Food Poisoning": ["Drink plenty of fluids", "Avoid solid food initially", "ORS solution", "Seek help if blood in stool"],
    "Heart Disease":  ["EMERGENCY: Call ambulance immediately", "Chew aspirin if available", "Do NOT exert yourself", "Lie down and stay calm"],
    "Bronchitis":     ["Rest well", "Drink warm fluids", "Humidifier helps", "Consult doctor for persistent cough"],
    "Dengue":         ["Immediate medical attention", "Monitor platelet count", "Stay hydrated", "Avoid aspirin/ibuprofen"],
    "Malaria":        ["Urgent medical consultation", "Take prescribed anti-malarial drugs", "Use mosquito nets", "Stay hydrated"],
    "Cholera":        ["Emergency medical care", "ORS continuously", "Antibiotics as prescribed", "Boil all water"],
    "Chickenpox":     ["Isolate to prevent spread", "Calamine lotion for itching", "Antivirals if prescribed", "Do not scratch"],
}

RISK_ACTIONS = {
    "Low":    "✅ You can manage this at home with basic care. Monitor symptoms for 2-3 days.",
    "Medium": "⚠️ Your symptoms suggest moderate risk. Schedule a doctor visit within 24-48 hours.",
    "High":   "🚨 HIGH RISK DETECTED. Please consult a doctor or visit a hospital IMMEDIATELY.",
}

# ── SQLite Setup ───────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            symptoms    TEXT,
            disease     TEXT,
            risk_level  TEXT,
            timestamp   TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "AI Health Risk Prediction API", "status": "running"})


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Return list of all symptom names."""
    return jsonify({"symptoms": SYMPTOM_COLS})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST body (JSON):
    {
      "patient_name": "John",
      "symptoms": {
        "fever": 1,
        "cough": 1,
        "headache": 0,
        ...
      }
    }
    """
    if not disease_model:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 503

    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "Missing 'symptoms' in request body"}), 400

    patient_name = data.get("patient_name", "Anonymous")
    symptoms_input = data["symptoms"]

    # Build feature vector in correct order
    feature_vector = [int(symptoms_input.get(col, 0)) for col in SYMPTOM_COLS]

    # Predict disease
    disease_pred_enc = disease_model.predict([feature_vector])[0]
    disease_proba    = disease_model.predict_proba([feature_vector])[0]
    disease_name     = disease_encoder.inverse_transform([disease_pred_enc])[0]

    # Predict risk level
    risk_pred_enc = risk_model.predict([feature_vector])[0]
    risk_name     = risk_encoder.inverse_transform([risk_pred_enc])[0]

    # Build top-3 disease probabilities
    top3_idx  = disease_proba.argsort()[-3:][::-1]
    top3      = [
        {"disease": disease_encoder.classes_[i], "probability": round(float(disease_proba[i]) * 100, 1)}
        for i in top3_idx
    ]

    # Recommendations
    recs       = RECOMMENDATIONS.get(disease_name, ["Consult a healthcare professional"])
    risk_msg   = RISK_ACTIONS.get(risk_name, "Please consult a doctor.")
    symptom_count = sum(feature_vector)

    # Emergency override: if chest pain + shortness of breath → force High
    if symptoms_input.get("chest_pain") and symptoms_input.get("shortness_of_breath"):
        risk_name = "High"
        risk_msg  = "🚨 EMERGENCY: Chest pain + shortness of breath detected. Call emergency services NOW!"

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (patient_name, symptoms, disease, risk_level, timestamp) VALUES (?,?,?,?,?)",
        (patient_name, json.dumps(symptoms_input), disease_name, risk_name, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    return jsonify({
        "patient_name":    patient_name,
        "predicted_disease": disease_name,
        "risk_level":      risk_name,
        "risk_message":    risk_msg,
        "recommendations": recs,
        "top_predictions": top3,
        "symptoms_reported": symptom_count,
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/history", methods=["GET"])
def get_history():
    """Return all past predictions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/history/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Record deleted"})


@app.route("/stats", methods=["GET"])
def stats():
    """Dashboard stats."""
    conn = sqlite3.connect(DB_PATH)
    total     = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    high_risk = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level='High'").fetchone()[0]
    diseases  = conn.execute(
        "SELECT disease, COUNT(*) as cnt FROM predictions GROUP BY disease ORDER BY cnt DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return jsonify({
        "total_predictions": total,
        "high_risk_count":   high_risk,
        "top_diseases":      [{"disease": d[0], "count": d[1]} for d in diseases],
    })


if __name__ == "__main__":
    print("\n🚀 AI Health Risk Prediction API")
    print("   URL: http://localhost:5000")
    print("   Make sure you ran: python model/train_model.py\n")
    app.run(debug=True, port=5000)
