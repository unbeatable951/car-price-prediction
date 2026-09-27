"""
config.py
=========
Centralized configuration for the Car Price Prediction project.

WHY THIS FILE EXISTS
---------------------
In a production ML project, hardcoding file paths, column names, or
hyperparameters inside every script is a maintenance nightmare. If the
dataset location or a column name changes, you'd have to hunt through
every file. Instead, every module (data_ingestion, preprocessing,
train, predict, Flask app) imports constants FROM HERE.

This is the "single source of truth" pattern used in real ML systems
(similar to how Django/Flask apps use settings.py).
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# BASE PATHS
# ----------------------------------------------------------------------
# BASE_DIR resolves to the project root regardless of which directory
# the script is executed FROM. This avoids the classic
# "works on my machine but breaks on the server" path bug.
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

# Ensure critical directories exist at import time so no module ever
# crashes with "FileNotFoundError: directory does not exist".
#
# Guarded with try/except: some deployment environments (a read-only
# container filesystem, a serverless platform, restrictive file
# permissions) can't create directories at import time. Previously,
# `import config` itself would raise and take down the entire process
# before any other code — including error handling — had a chance to
# run. Now a failure here is logged to stderr (the real logger isn't
# set up yet, since src/logging_utils.py itself imports this module)
# and import proceeds; anything that actually needs to write to a
# missing directory will fail with its own clear error at that point,
# rather than the whole app refusing to even start.
for _dir in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError as _dir_error:
        import sys as _sys

        print(
            f"WARNING: config.py could not create directory '{_dir}' "
            f"({_dir_error}). Continuing — code that writes to this "
            f"directory will raise its own error if it's actually needed.",
            file=_sys.stderr,
        )

# ----------------------------------------------------------------------
# DATA FILES
# ----------------------------------------------------------------------
RAW_DATA_FILE = DATA_DIR / "cardekho_dataset.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_car_data.csv"

# ----------------------------------------------------------------------
# MODEL / ARTIFACT FILES
# ----------------------------------------------------------------------
# We version the model filename with "latest" as a symlink-like pointer.
# See src/train.py for how MODEL_VERSION gets stamped in.
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

MODEL_FILE = MODELS_DIR / f"car_price_model_{MODEL_VERSION}.joblib"
LATEST_MODEL_FILE = MODELS_DIR / "car_price_model_latest.joblib"

PREPROCESSOR_FILE = ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_NAMES_FILE = ARTIFACTS_DIR / "feature_names.json"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"
MODEL_REGISTRY_FILE = ARTIFACTS_DIR / "model_registry.json"

# ----------------------------------------------------------------------
# DATASET SCHEMA
# ----------------------------------------------------------------------
# CONFIRMED against the actual file (data/cardekho_dataset.csv) via:
#   df.columns.tolist()
# -> ['Unnamed: 0', 'car_name', 'brand', 'model', 'vehicle_age',
#     'km_driven', 'seller_type', 'fuel_type', 'transmission_type',
#     'mileage', 'engine', 'max_power', 'seats', 'selling_price']
#
# This is the NEWER, extended CarDekho dataset — NOT the older 8-column
# version (which had Present_Price/Year/Owner). Notably:
#   - 'vehicle_age' is ALREADY provided (no need to derive it from Year).
#   - There is no 'owner' or 'present_price' column in this version.
#   - Several extra engineering-spec numeric columns exist: mileage
#     (km/l or km/kg), engine (CC), max_power (bhp), seats.
TARGET_COLUMN = "selling_price"

RAW_COLUMNS = [
    "Unnamed: 0",
    "car_name",
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "selling_price",
]

# Columns dropped during feature engineering and why:
#   - Unnamed: 0   : leftover pandas index column saved into the CSV,
#                    pure noise, zero predictive value.
#   - car_name     : full name string (e.g. "Maruti Alto LXi") — extremely
#                    high cardinality (near one-hot per row), would cause
#                    massive overfitting via one-hot encoding.
#   - model        : also very high cardinality (100+ unique models);
#                    'brand' captures most of the useful signal at a
#                    manageable cardinality (~30 brands), so we keep
#                    brand and drop the finer-grained model.
COLUMNS_TO_DROP = ["Unnamed: 0", "car_name", "model"]

# 'vehicle_age' is used directly (no Year-to-age conversion needed here,
# since this dataset already provides it pre-computed).
NUMERICAL_FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
]
CATEGORICAL_FEATURES = ["brand", "seller_type", "fuel_type", "transmission_type"]

# ----------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42  # fixed for reproducibility across runs

# ----------------------------------------------------------------------
# CROSS VALIDATION
# ----------------------------------------------------------------------
CV_FOLDS = 5

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "project.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# ----------------------------------------------------------------------
# FLASK APP CONFIG (used later by app/)
# ----------------------------------------------------------------------
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False") == "True"

# ----------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------
# Comma-separated list of allowed browser origins, e.g.:
#   ALLOWED_ORIGINS="https://valugauge.example.com,https://staging.example.com"
# Defaults to "*" (any origin) ONLY for local development convenience —
# every deployed environment (Render, etc.) should set ALLOWED_ORIGINS
# explicitly so the API can't be called cross-origin from arbitrary sites.
_allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    "*" if _allowed_origins_raw.strip() == "*" else [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
)

# ----------------------------------------------------------------------
# RATE LIMITING
# ----------------------------------------------------------------------
# Applied to POST /predict specifically (the only expensive/scrapable
# endpoint). Expressed as a flask-limiter rate string; override per
# environment via env var without a code change.
PREDICT_RATE_LIMIT = os.getenv("PREDICT_RATE_LIMIT", "30 per minute")