import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib, json
from pathlib import Path

# === [1] Path Setup ===
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "floq_reviews_clean.csv"
MODELS_DIR = ROOT_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
META_PATH = MODELS_DIR / "metadata.json"

# === [2] Load Dataset ===
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Hapus baris kosong atau NaN di clean_content
df.dropna(subset=["clean_content"], inplace=True)
df = df[df["clean_content"].str.strip() != ""]

print(f"✅ Loaded {len(df)} cleaned reviews after cleaning NaN/empty texts")


# === [3] Split Data ===
X = df["clean_content"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === [4] TF-IDF Vectorization ===
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# === [5] Train Logistic Regression ===
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train_vec, y_train)

# === [6] Evaluate Model ===
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print(f"\n🎯 Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === [7] Save Model, Vectorizer, and Metadata ===
joblib.dump(model, MODEL_PATH)
joblib.dump(tfidf, VECTORIZER_PATH)

metadata = {
    "model_name": "LogisticRegression + TF-IDF",
    "accuracy": acc,
    "dataset_size": len(df),
    "classes": df["sentiment"].unique().tolist(),
    "sample_counts": df["sentiment"].value_counts().to_dict(),
}
json.dump(metadata, open(META_PATH, "w"), indent=4)

print(f"\n💾 Model saved to: {MODEL_PATH}")
print(f"💾 Vectorizer saved to: {VECTORIZER_PATH}")
print(f"🧾 Metadata saved to: {META_PATH}")
