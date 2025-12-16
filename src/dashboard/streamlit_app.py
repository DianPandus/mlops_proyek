import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Floq Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"
META_PATH = Path("data/models/metadata.json")
DATA_PATH = Path("data/processed/floq_reviews_clean.csv")

# =========================================================
# LOAD METADATA
# =========================================================
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
META_PATH = ROOT_DIR / "models" / "metadata.json"

with open(META_PATH) as f:
    metadata = json.load(f)


# =========================================================
# HEADER
# =========================================================
st.title("📊 Floq Sentiment Analysis Dashboard")
st.markdown(
    """
    Dashboard ini digunakan untuk melakukan **analisis sentimen ulasan aplikasi Floq**  
    serta menampilkan **ringkasan performa model machine learning**.
    """
)

st.divider()

# =========================================================
# OVERVIEW METRICS
# =========================================================
st.subheader("📌 Project Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("📦 Dataset Size", f"{metadata['dataset_size']:,}")
col2.metric("🧠 Best Model", metadata["model_name"])
col3.metric("🎯 Accuracy", f"{metadata['accuracy']:.2%}")
col4.metric("🏷️ Classes", len(metadata["classes"]))

st.divider()

# =========================================================
# SENTIMENT PREDICTION
# =========================================================
st.subheader("🔮 Sentiment Prediction")

text_input = st.text_area(
    "Masukkan ulasan aplikasi:",
    placeholder="Contoh: aplikasinya sering error dan bikin kesel",
    height=120
)

if st.button("🚀 Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        response = requests.post(API_URL, json={"text": text_input})

        if response.status_code == 200:
            result = response.json()
            sentiment = result["predicted_sentiment"]

            if sentiment == "positif":
                st.success("🟢 Sentimen: **POSITIF**")
            elif sentiment == "negatif":
                st.error("🔴 Sentimen: **NEGATIF**")
            else:
                st.warning("🟡 Sentimen: **NETRAL**")
        else:
            st.error("❌ Gagal memanggil API")

st.divider()

# =========================================================
# SENTIMENT DISTRIBUTION
# =========================================================
st.subheader("📊 Distribusi Sentimen Dataset")

sent_df = pd.DataFrame(
    metadata["sample_counts"].items(),
    columns=["Sentiment", "Jumlah"]
)

st.bar_chart(sent_df.set_index("Sentiment"))

st.divider()

# =========================================================
# WORD CLOUD
# =========================================================
st.subheader("☁️ Word Cloud Ulasan")

df = pd.read_csv(DATA_PATH)
all_text = " ".join(df["clean_content"].dropna().astype(str))

wordcloud = WordCloud(
    width=1200,
    height=500,
    background_color="black",
    colormap="Set2"
).generate(all_text)

fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)

st.divider()

# =========================================================
# MODEL INFO
# =========================================================
st.subheader("🧠 Model Information")

st.markdown(
    f"""
    - **Model**: {metadata['model_name']}
    - **Feature Extraction**: TF-IDF (unigram + bigram)
    - **Evaluation Metric**: Accuracy
    - **Classes**: {", ".join(metadata['classes'])}
    """
)

st.info(
    "Pipeline model ini dilatih dan divalidasi secara otomatis "
    "menggunakan CI GitHub Actions untuk menjamin reproducibility."
)

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "📌 Floq Sentiment Analysis | Machine Learning & MLOps Project"
)
