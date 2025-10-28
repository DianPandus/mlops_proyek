# ===========================================================
# FLOQ SENTIMENT API DOCKERFILE
# ===========================================================

# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas joblib

# Expose port FastAPI
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
