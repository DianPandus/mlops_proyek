# 🚀 FLOQ Sentiment Analysis — MLOps Project (FastAPI + Docker)

Proyek ini merupakan implementasi **microservice Machine Learning berbasis Docker** yang bertujuan untuk melakukan analisis sentimen pada ulasan aplikasi **FLOQ** dari Google Playstore.  
Pipeline ini mencakup seluruh tahapan MLOps — mulai dari _data collection_, _ETL_, _model training_, _serving API_, hingga _containerization_.

---

## 🧩 1. Project Structure

```

MLOPS/
├── data/
│ ├── master/ # hasil scraping mentah
│ ├── processed/ # data setelah preprocessing
│ └── models/ # model + vectorizer hasil training
│
├── src/
│ ├── scrapping/
│ │ └── scrapping_dataset.py
│ ├── etl/
│ │ └── preprocess.py
│ ├── training/
│ │ └── train_sentiment.py
│ └── api/
│ └── app.py # FastAPI endpoint untuk prediksi
│
├── requirements.txt
├── Dockerfile
└── README.md

```

---

## ⚙️ 2. Setup Environment

Pastikan kamu sudah menginstall **Python 3.11** dan **Docker Desktop**.

### 📦 Install dependency

```bash
pip install -r requirements.txt
```

---

## 📊 3. Pipeline Workflow

### 🕵️‍♂️ (1) Scraping Data Review

Mengambil ulasan dari aplikasi FLOQ di Play Store.

```bash
cd src/scrapping
python scrapping_dataset.py
```

📂 Hasil: `data/master/floq_reviews_master.csv`

---

### 🧹 (2) ETL & Preprocessing

Membersihkan data hasil scraping, menghapus duplikat, menghilangkan karakter noise, dan menyimpan versi bersih.

```bash
cd ../etl
python preprocess.py
```

📂 Hasil: `data/processed/floq_reviews_clean.csv`

---

### 🧠 (3) Model Training

Melatih model **TF-IDF + Logistic Regression** untuk klasifikasi sentimen (positif, netral, negatif).

```bash
cd ../training
python train_sentiment.py
```

📂 Hasil:

- `data/models/sentiment_model.pkl`
- `data/models/vectorizer.pkl`
- `data/models/metadata.json`

---

### 🌐 (4) Serving API (FastAPI)

Menjalankan layanan API untuk melakukan prediksi sentimen berbasis teks.

```bash
cd ../api
python -m uvicorn src.api.app:app --reload
```

Akses dokumentasi Swagger UI:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Contoh input di `/predict`:

```json
{ "text": "Aplikasi Floq sangat membantu dan mudah digunakan!" }
```

Response:

```json
{
  "input": "Aplikasi Floq sangat membantu dan mudah digunakan!",
  "predicted_sentiment": "positif"
}
```

---

## 🐳 (5) Deployment with Docker

### 🔹 Build Docker Image

Pastikan kamu berada di **root folder (MLOPS/)**
Jalankan perintah berikut:

```bash
docker build -t floq-sentiment-api .
```

### 🔹 Jalankan Container

```bash
docker run -d -p 8000:8000 --name floq-api floq-sentiment-api
```

📍 API berjalan di:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 🔹 Lihat container yang berjalan

```bash
docker ps
```

### 🔹 Stop container (jika sudah selesai)

```bash
docker stop floq-api
docker rm floq-api
```

---

## 🧠 Model Overview

| Komponen          | Algoritma                      | Akurasi |
| ----------------- | ------------------------------ | ------- |
| Model             | Logistic Regression            | ± 0.88  |
| Feature Extractor | TF-IDF (1-2 ngram, 5000 fitur) | -       |
| Label             | Positif, Netral, Negatif       | -       |

---

## 📦 Tech Stack

- **Python 3.11**
- **FastAPI + Uvicorn**
- **scikit-learn**
- **pandas, numpy, nltk**
- **Docker**
- _(Opsional)_ MLflow & DVC untuk tracking pipeline

---

## 🧱 Dockerfile Overview

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🧩 Flow Summary

```text
[Scraping] → [ETL Preprocessing] → [Model Training] → [FastAPI Serving] → [Dockerized API]
```
