# ===========================================================
#  TRAINING MULTI-MODEL SENTIMENT ANALYSIS
#  Models: Logistic Regression, XGBoost, SVM
#  FLOQ MLOps Project - Final Version
# ===========================================================

import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# === [1] Path Setup ===
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "floq_reviews_clean.csv"
MODELS_DIR = ROOT_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
LOGREG_PATH = MODELS_DIR / "logreg_model.pkl"
XGB_PATH = MODELS_DIR / "xgb_model.pkl"
SVM_PATH = MODELS_DIR / "svm_model.pkl"
TFIDF_PATH = MODELS_DIR / "vectorizer.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
META_PATH = MODELS_DIR / "metadata.json"

# === [2] Load Dataset ===
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["clean_content"], inplace=True)
df = df[df["clean_content"].str.strip() != ""]
print(f"📊 Dataset loaded: {len(df)} valid reviews")

# === [3] Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])
joblib.dump(label_encoder, ENCODER_PATH)

X = df["clean_content"]

# === [4] Data Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === [5] TF-IDF Vectorization ===
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
joblib.dump(tfidf, TFIDF_PATH)

# === [6] Train Logistic Regression ===
logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
logreg.fit(X_train_vec, y_train)
logreg_pred = logreg.predict(X_test_vec)
logreg_acc = accuracy_score(y_test, logreg_pred)
print(f"\n=== Logistic Regression ===\nAccuracy: {logreg_acc:.4f}")
print(classification_report(y_test, logreg_pred, target_names=label_encoder.classes_))
joblib.dump(logreg, LOGREG_PATH)

# === [7] Train XGBoost ===
xgb_model = xgb.XGBClassifier(
    eval_metric="mlogloss",
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8
)
xgb_model.fit(X_train_vec, y_train)
xgb_pred = xgb_model.predict(X_test_vec)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"\n=== XGBoost ===\nAccuracy: {xgb_acc:.4f}")
print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))
joblib.dump(xgb_model, XGB_PATH)

# === [8] Train SVM ===
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"\n=== SVM (LinearSVC) ===\nAccuracy: {svm_acc:.4f}")
print(classification_report(y_test, svm_pred, target_names=label_encoder.classes_))
joblib.dump(svm_model, SVM_PATH)

# === [9] Save Metadata ===
metadata = {
    "dataset_size": len(df),
    "label_classes": label_encoder.classes_.tolist(),
    "models": {
        "logreg": float(logreg_acc),
        "xgb": float(xgb_acc),
        "svm": float(svm_acc)
    },
    "best_model": max(
        [("logreg", logreg_acc), ("xgb", xgb_acc), ("svm", svm_acc)],
        key=lambda x: x[1]
    )[0]
}

json.dump(metadata, open(META_PATH, "w"), indent=4)

print("\n📌 FINAL ACCURACY")
for model, acc in metadata["models"].items():
    print(f" - {model.upper()} → {acc:.4f}")

print(f"\n🔥 BEST MODEL SELECTED → {metadata['best_model'].upper()}")
print(f"💾 Metadata saved: {META_PATH}")
print("🎉 Training pipeline complete!")
