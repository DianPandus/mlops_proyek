import pandas as pd
import re, os
from pathlib import Path

# === [1] Path Setup ===
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MASTER_PATH = DATA_DIR / "master" / "floq_reviews_master.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "floq_reviews_clean.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Load master data from: {MASTER_PATH}")

# === [2] Load data master ===
if not MASTER_PATH.exists():
    print("❌ File master tidak ditemukan. Pastikan sudah menjalankan scrapping_dataset.py dulu.")
    exit()

df = pd.read_csv(MASTER_PATH)

if df.empty:
    print("⚠️ File master kosong. Tidak ada data untuk diproses.")
    exit()

print(f"✅ Loaded {len(df)} rows.")

# === [3] Cleaning ===
df.drop_duplicates(subset=["reviewId"], inplace=True)
df.dropna(subset=["content"], inplace=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_content"] = df["content"].apply(clean_text)

# === [4] Label Sentiment ===
def label_sentiment(score):
    if score >= 4:
        return "positif"
    elif score == 3:
        return "netral"
    else:
        return "negatif"

df["sentiment"] = df["score"].apply(label_sentiment)

# === [5] Save Clean Data ===
df.to_csv(PROCESSED_PATH, index=False)
print(f"💾 Data berhasil disimpan ke: {PROCESSED_PATH}")
print(f"Total baris: {len(df)}")

# === [6] Tampilkan contoh hasil ===
print("\n🔍 Preview data bersih:")
print(df[["score", "content", "clean_content", "sentiment"]].head(5))
