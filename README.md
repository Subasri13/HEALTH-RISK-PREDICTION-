# 🩺 MediAI – AI Health Risk Prediction System
### Complete Project | Python + Flask + ML + HTML Frontend

---

## 📁 Project Structure

```
health_ai/
├── dataset/
│   └── symptom_disease_dataset.csv    ← Training dataset (100 records, 15 symptoms)
├── model/
│   └── train_model.py                 ← Train & save ML models
├── backend/
│   ├── app.py                         ← Flask REST API
│   ├── requirements.txt               ← Python dependencies
│   └── models/                        ← (auto-created after training)
│       ├── disease_model.pkl
│       ├── risk_model.pkl
│       ├── disease_encoder.pkl
│       ├── risk_encoder.pkl
│       └── symptoms.json
└── frontend/
    └── index.html                     ← Complete web UI (open in browser)
```

---

## 🚀 Setup Instructions (Step by Step)

### Step 1 – Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 – Train the ML models
```bash
cd model
python train_model.py
```
✅ This creates `backend/models/` with all saved model files.

### Step 3 – Start the Flask API
```bash
cd backend
python app.py
```
API runs at: **http://localhost:5000**

### Step 4 – Open the Frontend
Open `frontend/index.html` in any browser.
- Works standalone (demo mode) without the backend
- Connects automatically to Flask if it's running

---

## 🌐 API Endpoints

| Method | Endpoint         | Description                    |
|--------|-----------------|-------------------------------|
| GET    | `/`             | API health check               |
| GET    | `/symptoms`     | List all 15 symptoms           |
| POST   | `/predict`      | Predict disease + risk level   |
| GET    | `/history`      | Get all past predictions       |
| GET    | `/stats`        | Dashboard statistics           |
| DELETE | `/history/<id>` | Delete a prediction record     |

### Sample POST /predict request:
```json
{
  "patient_name": "Arjun Kumar",
  "symptoms": {
    "fever": 1,
    "cough": 1,
    "headache": 1,
    "fatigue": 1,
    "shortness_of_breath": 1
  }
}
```

### Sample Response:
```json
{
  "predicted_disease": "COVID-19",
  "risk_level": "High",
  "risk_message": "🚨 HIGH RISK DETECTED. Please consult a doctor IMMEDIATELY.",
  "recommendations": ["Isolate immediately", "Monitor SpO2", "Get tested"],
  "top_predictions": [
    {"disease": "COVID-19", "probability": 87.5},
    {"disease": "Influenza", "probability": 62.0},
    {"disease": "Pneumonia", "probability": 41.0}
  ],
  "symptoms_reported": 5,
  "timestamp": "2024-10-15 14:32:00"
}
```

---

## 🧠 ML Model Details

| Model           | Algorithm         | Task                    |
|----------------|------------------|------------------------|
| Disease Model  | Random Forest     | Multi-class prediction |
| Risk Model     | Random Forest     | 3-class: Low/Med/High  |

**15 Input Features (Symptoms):**
fever, cough, headache, fatigue, nausea, chest_pain,
shortness_of_breath, sore_throat, body_ache, diarrhea,
vomiting, runny_nose, dizziness, rash, loss_of_appetite

**12 Diseases Predicted:**
Common Cold, Influenza, COVID-19, Migraine, Gastroenteritis,
Pneumonia, Allergy, Sinusitis, Typhoid, Food Poisoning,
Heart Disease, Bronchitis, Dengue, Malaria, Cholera, Chickenpox

---

## 🎯 Features

- ✅ Symptom checkboxes with severity selector
- ✅ Disease prediction with probability bars
- ✅ Risk classification (Low / Medium / High)
- ✅ Personalized recommendations per disease
- ✅ Emergency alert for chest pain + breathlessness
- ✅ Prediction history with localStorage
- ✅ SQLite database for backend storage
- ✅ REST API (Flask + CORS)
- ✅ Works in demo mode without backend

---

## 🎤 Interview Answer

> "This project is an AI-based health risk prediction system that analyzes patient symptoms using machine learning to predict possible diseases and risk levels. It uses Random Forest classifiers trained on a symptom-disease dataset with 15 features. The system provides intelligent recommendations, emergency alerts, and stores prediction history in a database. The frontend connects to a Flask REST API and works in demo mode without backend dependency."

---

## ⚠️ Disclaimer
This tool is for educational purposes only and is NOT a substitute for professional medical advice.
