# ===========================================================
# FASTAPI SENTIMENT API (LAZY LOAD - CI SAFE)
# ===========================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, json
from pathlib import Path

app = FastAPI(
    title="Floq Sentiment API",
    description="Prediksi Sentimen Ulasan (Auto Best Model)",
    version="2.1.0"
)

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"

# === GLOBAL CACHE ===
model = None
vectorizer = None
label_encoder = None
metadata = None
best_model = None


def load_artifacts():
    global model, vectorizer, label_encoder, metadata, best_model

    if model is not None:
        return  # already loaded

    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError("metadata.json not found")

    with open(meta_path) as f:
        metadata = json.load(f)

    if "best_model" not in metadata:
        raise RuntimeError("metadata.json missing 'best_model'")

    best_model = metadata["best_model"]

    model_path = MODELS_DIR / f"{best_model}_model.pkl"
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

    print(f"🚀 Loaded model: {best_model.upper()}")


class ReviewInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "Sentiment API is running 🚀",
        "status": "OK"
    }


@app.post("/predict")
def predict(input_data: ReviewInput):
    try:
        load_artifacts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    X = vectorizer.transform([input_data.text])
    pred_num = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    return {
        "input": input_data.text,
        "predicted_sentiment": pred_label,
        "model_used": best_model
    }
