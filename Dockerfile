# ============================================================
# Dockerfile — Car Price Prediction API
# ============================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Cleaned up: just standard build tools and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This copies everything over
COPY . .

EXPOSE 5000
ENV PORT=5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Change --workers 2 to --workers 1
CMD gunicorn app.app:app --workers 1 --bind 0.0.0.0:$PORT --timeout 60