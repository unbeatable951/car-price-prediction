# ============================================================
# Dockerfile — Car Price Prediction API
# ============================================================
# Base image: python:3.12-slim
# "slim" (not full python:3.12) keeps the image small — it excludes
# compilers/docs/etc. we don't need at runtime, which means faster
# builds, faster deploys, and a smaller attack surface.
FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr —
# without this, logs can appear delayed or out of order in `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory inside the container — all subsequent COPY/RUN
# commands are relative to this.
WORKDIR /app

# Install OS-level build dependencies needed to compile some Python
# packages (e.g. some numpy/scipy/xgboost/catboost wheels still need
# a C compiler on certain platforms), plus curl for the HEALTHCHECK
# below. Removed in the same layer via rm -rf to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements.txt first (not the whole project). Docker
# caches layers — as long as requirements.txt doesn't change, Docker
# reuses the cached "pip install" layer on rebuilds, even if you've
# changed application code. This makes iterative rebuilds much faster.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project.
COPY . .

# Document which port the container listens on. This is informational
# for anyone reading the Dockerfile / `docker inspect` — it does NOT
# actually publish the port (that happens via `docker run -p` or
# Render's port configuration).
EXPOSE 5000

# Render (and most PaaS platforms) inject a $PORT environment variable
# at runtime and expect the app to bind to it — defaulting to 5000
# for local `docker run` where $PORT isn't set.
ENV PORT=5000

# Basic container health check — Docker will mark the container
# "unhealthy" if this fails repeatedly, which orchestrators (Docker
# Swarm, Kubernetes, Render) use to decide whether to restart it.
# Uses curl (not python -c) because shell-form CMD with a Python
# one-liner containing nested quotes is fragile across shells — curl
# is the standard, simple choice for container health checks.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Run with Gunicorn (production WSGI server) instead of Flask's own
# dev server (`flask run` / `app.run()`), which is single-threaded,
# not production-hardened, and explicitly warns against production use.
#   - "app.app:app"  -> module app/app.py, WSGI object named `app`
#   - --workers 2    -> 2 worker processes (reasonable default for a
#                       small-to-medium deployment; tune based on CPU)
#   - --bind 0.0.0.0:$PORT -> listen on all interfaces, on whatever
#                       port the platform assigns via $PORT
#   - --timeout 60   -> generous timeout since scikit-learn model
#                       loading/prediction can take longer than
#                       Gunicorn's 30s default on first request
CMD gunicorn app.app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 60