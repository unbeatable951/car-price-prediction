# ============================================================
# Dockerfile — Car Price Prediction API
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- UPDATE: Added git and git-lfs into the apt-get layer ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files (including the LFS pointer)
COPY . .

# --- NEW: Force Git LFS to pull down the REAL 122MB joblib file ---
# We use a dummy git initialization because the docker build context 
# strips the hidden .git directory required by git-lfs.
RUN git init && \
    git remote add origin https://github.com/unbeatable951/car-price-prediction.git && \
    git fetch --depth=1 origin main && \
    git lfs pull

EXPOSE 5000
ENV PORT=5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

CMD gunicorn app.app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 60