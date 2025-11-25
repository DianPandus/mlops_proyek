# ===========================================================
# FASTAPI SENTIMENT API (AUTO-LOAD BEST MODEL)
# ===========================================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
from pathlib import Path

app = FastAPI(
    title="Floq Sentiment API",
    description="Prediksi Sentimen Ulasan (Model Otomatis Terbaik)",
    version="2.0.0"
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "data" / "models"

# Load metadata
metadata = json.load(open(MODELS_DIR / "metadata.json"))
best_model = metadata["best_model"]

# Load model, vectorizer, and encoder
model = joblib.load(MODELS_DIR / f"{best_model}_model.pkl")
vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

print(f"🚀 Loaded best model → {best_model.upper()}")

class ReviewInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "message": "Sentiment API is running 🚀",
        "best_model": best_model,
        "classes": metadata["label_classes"],
    }

@app.post("/predict")
def predict(input_data: ReviewInput):
    X = vectorizer.transform([input_data.text])
    pred_num = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    return {
        "input": input_data.text,
        "predicted_sentiment": pred_label,
        "model_used": best_model
    }
