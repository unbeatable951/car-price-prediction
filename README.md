# 🚗 Car Price Prediction — Production ML Project

An end-to-end, production-grade machine learning system that predicts
the resale value of used cars, trained on the CarDekho dataset.
Includes a full ML pipeline (ingestion → EDA → feature engineering →
preprocessing → training → evaluation → tuning), a Flask REST API,
a custom frontend, Docker containerization, and Render deployment —
built the way a real ML engineering team would structure it.

---

## Architecture

```
                     ┌─────────────────────┐
                     │   cardekho_dataset  │
                     │        .csv         │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼───────────┐
                     │   data_ingestion.py  │  loads + validates schema
                     └──────────┬───────────┘
                                │
              ┌─────────────────▼────────────────── ┐
              │   sklearn Pipeline (per model)      │
              │  ┌────────────────────────────────┐ │
              │  │  FeatureEngineer               │ │  drop cols, impute
              │  │  (feature_engineering.py)      │ │  (fit on train only)
              │  └────────────────┬───────────────┘ │
              │  ┌────────────────▼───────────────┐ │
              │  │  Preprocessor                  │ │  encode + scale
              │  │  (preprocessing.py)            │ │  (fit on train only)
              │  └────────────────┬───────────────┘ │
              │  ┌────────────────▼───────────────┐ │
              │  │  Regressor (1 of 9 candidates) │ │
              │  └────────────────────────────────┘ │
              └─────────────────┬───────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │      train.py        │  trains all 9 + CV
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │     evaluate.py      │  compares metrics,
                     │                      │  selects + tunes winner,
                     │                      │  saves production model
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │  models/*.joblib     │  versioned model registry
                     └──────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────----
        │                       │                          │
┌───────▼────────┐     ┌──────────▼────────┐      ┌────────▼────────┐
│  src/predict.py│     │   app/ (Flask API)│      │  pytest suite   │
│  CLI/notebook  │     │  GET  /           │      │  (tests/)       │
│  interface     │     │  GET  /health     │      └─────────────────┘
└────────────────┘     │  POST /predict    │
                       └──────────┬────────┘
                                  │
                       ┌──────────▼──────────┐
                       │ templates/ + static │  Bootstrap frontend
                       │(fetch() -> /predict)│
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │  Docker + Gunicorn  │
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │   Render (deployed) │
                       └─────────────────────┘
```

**Key design decision:** every candidate model is wrapped in an identical
`Pipeline([FeatureEngineer, Preprocessor, Model])`. This means feature
engineering and preprocessing statistics (medians, encoder categories,
scaler mean/std) are *always* fit on training data only, and the entire
pipeline — not just the model — is what gets saved to disk. The Flask
API loads one file and calls `.predict()` on a raw dataframe; it never
needs its own copy of preprocessing logic, eliminating train/serve skew.

---

## Project Structure

```
car-price-prediction/
├── app/                        # Flask backend + frontend
│   ├── app.py                  # App factory, Gunicorn entrypoint
│   ├── routes.py                # Blueprint: /, /health, /predict
│   ├── predict.py                # Pydantic schema + predictor singleton
│   ├── templates/index.html
│   └── static/{css,js}/
├── src/                        # Core ML pipeline
│   ├── data_ingestion.py
│   ├── feature_engineering.py   # Custom sklearn transformer
│   ├── preprocessing.py         # Custom sklearn transformer
│   ├── train.py                 # Trains 9 models + CV
│   ├── evaluate.py               # Compares, tunes, saves best model
│   └── predict.py                # CarPricePredictor class
├── notebooks/eda.ipynb
├── tests/                       # pytest suite (23 tests)
├── data/                        # cardekho_dataset.csv (not committed)
├── models/                      # versioned .joblib models (not committed)
├── artifacts/                   # preprocessor, metrics, registry (not committed)
├── logs/project.log
├── .github/workflows/ci-cd.yml  # lint + test + docker build on push
├── config.py                    # single source of truth for paths/schema
├── requirements.txt
├── Dockerfile / .dockerignore
├── render.yaml
├── .pre-commit-config.yaml
└── README.md
```

---

## Setup — Local Development

### 1. Clone and enter the project
```bash
cd car-price-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the CarDekho dataset from Kaggle and place it at:
```
data/cardekho_dataset.csv
```
Expected columns: `car_name, brand, model, vehicle_age, km_driven,
seller_type, fuel_type, transmission_type, mileage, engine, max_power,
seats, selling_price`.

### 5. Run the ML pipeline
```bash
python -m src.data_ingestion    # sanity check the dataset loads
python -m src.train              # trains all 9 candidate models
python -m src.evaluate           # compares, tunes, saves the best one
```
This produces `models/car_price_model_latest.joblib` and
`artifacts/model_comparison.csv`.

### 6. Run the API locally
```bash
python app/app.py
```
Visit `http://localhost:5000` in your browser.

### 7. Run tests
```bash
pytest tests/ -v
```

### 8. Enable pre-commit hooks (optional but recommended)
```bash
pip install pre-commit
pre-commit install
```

---

## API Documentation

### `GET /`
Returns basic service info.
```json
{
  "service": "Car Price Prediction API",
  "status": "running",
  "endpoints": { "...": "..." }
}
```

### `GET /health`
Health check — confirms the model loaded successfully.
- `200` → `{"status": "healthy", "model_loaded": true}`
- `503` → `{"status": "unhealthy", "model_loaded": false, "error": "..."}`

### `POST /predict`
Predicts a car's selling price.

**Request body:**
```json
{
  "brand": "Maruti",
  "vehicle_age": 5,
  "km_driven": 45000,
  "seller_type": "Individual",
  "fuel_type": "Petrol",
  "transmission_type": "Manual",
  "mileage": 18.5,
  "engine": 1200,
  "max_power": 85,
  "seats": 5
}
```

**Success response — `200`:**
```json
{ "predicted_price": 6.72 }
```

**Validation error — `422`** (bad type, out-of-range value, invalid category):
```json
{
  "error": "Invalid input data",
  "details": [
    {"field": "fuel_type", "message": "fuel_type must be one of [...], got 'Rocket Fuel'"}
  ]
}
```

**Malformed request — `400`:** missing/invalid JSON body.
**Server error — `500`:** unexpected prediction failure.
**Model unavailable — `503`:** returned by `/health` if the model failed to load.

---

## Docker

### Build
```bash
docker build -t car-price-prediction .
```

### Run
```bash
docker run -p 5000:5000 car-price-prediction
```
Visit `http://localhost:5000`.

### Test the containerized API
```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"brand":"Maruti","vehicle_age":5,"km_driven":45000,"seller_type":"Individual","fuel_type":"Petrol","transmission_type":"Manual","mileage":18.5,"engine":1200,"max_power":85,"seats":5}'
```

---

## Deployment (Render)

1. Push this repository to GitHub.
2. In Render: **New +** → **Blueprint** → connect your GitHub repo.
3. Render detects `render.yaml` automatically and provisions the service.
4. Every push to `main` triggers an automatic redeploy (`autoDeploy: true`).
5. View logs: Render dashboard → your service → **Logs** tab (streams in real time).

**Common deployment errors:**
| Error | Fix |
|---|---|
| Build fails on `pip install` | Check `requirements.txt` for a typo'd package/version |
| Health check fails, service restarts loop | Model file missing — ensure training ran and `models/` isn't excluded incorrectly |
| `Address already in use` | Don't hardcode port 5000 — the app already reads `$PORT` via `config.py` |
| 502 Bad Gateway right after deploy | Gunicorn still starting up — Render's health check has a grace period, wait ~30s |

---

## Model Performance

Model comparison and the selected model's justification are written to:
- `artifacts/model_comparison.csv` — full metrics table for all 9 models
- `artifacts/best_model_selection.json` — winning model, tuned hyperparameters, final test metrics
- `artifacts/model_registry.json` — version history of every trained/saved model

---

## Tech Stack
Python 3.12 · scikit-learn · XGBoost · CatBoost · Flask · Gunicorn ·
Pydantic · Bootstrap 5 · Docker · Render · GitHub Actions · pytest

---

## License
For educational/portfolio use.
