# ===========================================================
#  TRAINING MULTI-MODEL SENTIMENT ANALYSIS
#  Models: Logistic Regression, XGBoost, SVM
#  FLOQ MLOps Project - Final Version (HARDENED)
# ===========================================================

import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

import xgboost as xgb

# === [1] Path Setup ===
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "floq_reviews_clean.csv"
MODELS_DIR = ROOT_DIR / "models"   # ⬅️ DISAMAKAN DENGAN DASHBOARD
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOGREG_PATH = MODELS_DIR / "logreg_model.pkl"
XGB_PATH = MODELS_DIR / "xgb_model.pkl"
SVM_PATH = MODELS_DIR / "svm_model.pkl"
TFIDF_PATH = MODELS_DIR / "vectorizer.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
META_PATH = MODELS_DIR / "metadata.json"

# === [2] Load Dataset ===
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["clean_content", "sentiment"], inplace=True)
df = df[df["clean_content"].str.strip() != ""]
sample_counts = df["sentiment"].value_counts().to_dict()
print(f"📊 Dataset loaded: {len(df)} valid reviews")

# === [3] Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])
joblib.dump(label_encoder, ENCODER_PATH)

X = df["clean_content"]

# === [4] Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# === [5] TF-IDF Vectorization ===
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
joblib.dump(tfidf, TFIDF_PATH)

# ===========================================================
# MODEL TRAINING
# ===========================================================

results = {}

# === [6] Logistic Regression ===
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_vec, y_train)
pred = logreg.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
results["logreg"] = acc

print(f"\n=== Logistic Regression ===")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, pred, target_names=label_encoder.classes_))
joblib.dump(logreg, LOGREG_PATH)

# === [7] XGBoost ===
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train_vec, y_train)
pred = xgb_model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
results["xgb"] = acc

print(f"\n=== XGBoost ===")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, pred, target_names=label_encoder.classes_))
joblib.dump(xgb_model, XGB_PATH)

# === [8] SVM ===
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
pred = svm_model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
results["svm"] = acc

print(f"\n=== SVM (LinearSVC) ===")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, pred, target_names=label_encoder.classes_))
joblib.dump(svm_model, SVM_PATH)

# === [8.5] Determine Best Model (BASED ON ACCURACY) ===
best_model_name, best_acc = max(results.items(), key=lambda x: x[1])


# ===========================================================
# METADATA (DASHBOARD + API FRIENDLY)
# ===========================================================

best_model = max(results.items(), key=lambda x: x[1])

# === [9] Save Metadata ===
metadata = {
    "model_name": best_model_name,
    "accuracy": float(best_acc),
    "all_models": {
    "logreg": float(results["logreg"]),
    "xgb": float(results["xgb"]),
    "svm": float(results["svm"])
    },
    "vectorizer": "tfidf",
    "num_features": 5000,
    "dataset_size": len(df),
    "classes": label_encoder.classes_.tolist(),
    "sample_counts": sample_counts,
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)


# === [10] Final Log ===
print("\n📌 FINAL ACCURACY")
for model, acc in results.items():
    print(f" - {model.upper()} → {acc:.4f}")

print(f"\n🔥 BEST MODEL → {metadata['model_name'].upper()}")
print(f"💾 Metadata saved to {META_PATH}")
print("🎉 Training pipeline completed successfully!")

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

if metadata["model_name"] == "logreg":
    joblib.dump(logreg, BEST_MODEL_PATH)
elif metadata["model_name"] == "xgb":
    joblib.dump(xgb_model, BEST_MODEL_PATH)
elif metadata["model_name"] == "svm":
    joblib.dump(svm_model, BEST_MODEL_PATH)

