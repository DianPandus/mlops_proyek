

# 🚀 Floq: MLOps Sentiment Analysis Pipeline

Floq adalah proyek MLOps *end-to-end* yang dirancang untuk melakukan penarikan data ulasan (scraping), pemrosesan data (ETL), pelatihan model machine learning secara otomatis, hingga penyajian data melalui Dashboard interaktif dan API.

## 🌐 Live Demo & Registry

* **Dashboard:** [floqdashboard.streamlit.app](https://floqdashboard.streamlit.app/)
* **Docker Image:** `dianpandus/floq:latest` (FastAPI)

---

## 🏗️ Arsitektur Proyek

Proyek ini mengimplementasikan siklus hidup MLOps yang otomatis:

1. **Data Ingestion:** Scraping ulasan aplikasi secara berkala dari Google Play Store.
2. **ETL Pipeline:** Pembersihan teks dan transformasi data menggunakan Python dan Pandas.
3. **Automated Training:** Melatih berbagai model (XGBoost, Logistic Regression, SVM) untuk menemukan performa terbaik.
4. **CI/CD Pipeline:** Menggunakan GitHub Actions untuk menjalankan otomatisasi setiap minggu atau setiap kali ada perubahan kode.
5. **Deployment:**
* **Dashboard:** Otomatis dideploy ke Streamlit Cloud.
* **API:** Containerized menggunakan Docker untuk serving model.



---

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Libraries:** Scikit-learn, XGBoost, Pandas, NLTK, WordCloud, Plotly.
* **Dashboard:** Streamlit.
* **API:** FastAPI & Uvicorn.
* **MLOps & Tools:** GitHub Actions (CI/CD), Docker, DVC, MLflow.

---

## 📁 Struktur Direktori

```text
.
├── .github/workflows/     # Konfigurasi GitHub Actions (CI/CD)
├── data/
│   ├── raw/               # Data mentah hasil scraping
│   └── processed/         # Data bersih siap training
├── docker/
│   ├── Dockerfile.api     # Docker untuk FastAPI
│   └── Dockerfile.dashboard # Docker untuk Dashboard
├── models/                # Artefak model (.pkl) dan metadata
├── src/
│   ├── scrapping/         # Script penarikan data
│   ├── etl/               # Script preprocessing data
│   ├── training/          # Script pelatihan model
│   ├── api/               # Source code FastAPI
│   └── dashboard/         # Source code Streamlit
├── packages.txt           # Dependensi sistem (Linux)
└── requirements.txt       # Dependensi Python

```

---

## ⚙️ Instalasi Lokal

1. **Clone Repository:**
```bash
git clone https://github.com/DianPandus/mlops_proyek.git
cd mlops_proyek

```


2. **Install Dependensi:**
```bash
pip install -r requirements.txt

```


3. **Jalankan Dashboard:**
```bash
streamlit run src/dashboard/streamlit_app.py

```



---

## 🤖 Otomatisasi CI/CD

Workflow `.github/workflows/ci.yml` menangani:

* **Scheduler:** Scraping dan retraining otomatis setiap hari Senin.
* **Write Permission:** Mengunggah kembali data dan model terbaru ke GitHub untuk memperbarui Dashboard secara *real-time*.
* **Docker Build:** Memperbarui image `dianpandus/floq:latest` di Docker Hub setiap kali ada push ke branch `main`.

---
