# ===========================================================
# FASTAPI SENTIMENT SERVICE (FLOQ)
# ===========================================================
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

# === [1] INIT APP ===
app = FastAPI(
    title="Floq Sentiment API",
    description="Microservice untuk prediksi sentimen ulasan aplikasi Floq",
    version="1.0.0"
)

# === [2] LOAD MODEL & VECTORIZER ===
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "data" / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "data" / "models" / "vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# === [3] INPUT SCHEMA ===
class ReviewInput(BaseModel):
    text: str

# === [4] ROUTES ===
@app.get("/")
def home():
    return {"message": "Welcome to Floq Sentiment API 👋"}

@app.post("/predict")
def predict_sentiment(input_data: ReviewInput):
    text = input_data.text
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    return {
        "input": text,
        "predicted_sentiment": prediction
    }

# === [5] LOCAL RUN ===
# Jalankan: uvicorn src.api.app:app --reload
