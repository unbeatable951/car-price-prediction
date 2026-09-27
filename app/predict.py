"""
app/predict.py
================
Bridges the Flask HTTP layer and the ML pipeline (src/predict.py).

RESPONSIBILITIES OF THIS FILE (and nothing else):
  1. Define the strict input schema (Pydantic) for what a valid
     prediction request looks like.
  2. Provide a single, lazily-loaded CarPricePredictor instance shared
     across requests (loading a joblib model from disk on every
     request would be needlessly slow).
  3. Translate ML-layer exceptions (PredictionError) into a form
     routes.py can turn into clean HTTP responses.

WHY PYDANTIC HERE, ON TOP OF src/predict.py's OWN VALIDATION
------------------------------------------------------------------
src/predict.py's `_validate_input()` only checks that required fields
are PRESENT. It does NOT check that "vehicle_age" is actually a number,
that "fuel_type" is one of the fuel types the model was trained on, or
that km_driven isn't negative. That's exactly what Pydantic gives us:
type coercion, range constraints, and enum-style validation, with
automatic, detailed error messages — before bad data ever reaches the
model. This is "defense in depth": Pydantic catches malformed HTTP
input; src/predict.py's checks catch anything that slips through when
called directly (e.g. from a test or notebook, bypassing Flask).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import CarPricePredictor  # noqa: E402

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Input schema
# ----------------------------------------------------------------------
class CarPredictionRequest(BaseModel):
    """
    Strict schema for a single prediction request. Field constraints
    are set to realistic real-world ranges — not arbitrary — so a
    request like km_driven=-500 or vehicle_age=200 is rejected with a
    clear 422 error instead of silently producing a nonsense prediction.
    """

    brand: str = Field(..., min_length=1, description="Car brand, e.g. 'Maruti'")
    vehicle_age: int = Field(..., ge=0, le=50, description="Age of the vehicle in years")
    km_driven: int = Field(..., ge=0, le=1_000_000, description="Total kilometers driven")
    seller_type: str = Field(..., description="'Individual', 'Dealer', or 'Trustmark Dealer'")
    fuel_type: str = Field(..., description="'Petrol', 'Diesel', 'CNG', 'LPG', or 'Electric'")
    transmission_type: str = Field(..., description="'Manual' or 'Automatic'")
    mileage: float = Field(..., ge=0, le=50, description="Mileage in km/l or km/kg")
    engine: float = Field(..., gt=0, le=10_000, description="Engine displacement in CC")
    max_power: float = Field(..., gt=0, le=2_000, description="Max power in bhp")
    seats: int = Field(..., ge=2, le=14, description="Number of seats")

    @field_validator("seller_type")
    @classmethod
    def validate_seller_type(cls, v: str) -> str:
        allowed = {"individual", "dealer", "trustmark dealer"}
        if v.strip().lower() not in allowed:
            raise ValueError(f"seller_type must be one of {sorted(allowed)}, got '{v}'")
        return v.strip()

    @field_validator("fuel_type")
    @classmethod
    def validate_fuel_type(cls, v: str) -> str:
        allowed = {"petrol", "diesel", "cng", "lpg", "electric"}
        if v.strip().lower() not in allowed:
            raise ValueError(f"fuel_type must be one of {sorted(allowed)}, got '{v}'")
        return v.strip()

    @field_validator("transmission_type")
    @classmethod
    def validate_transmission_type(cls, v: str) -> str:
        allowed = {"manual", "automatic"}
        if v.strip().lower() not in allowed:
            raise ValueError(f"transmission_type must be one of {sorted(allowed)}, got '{v}'")
        return v.strip()


# ----------------------------------------------------------------------
# Shared predictor instance (loaded once, reused across requests)
# ----------------------------------------------------------------------
_predictor: Optional[CarPricePredictor] = None


def get_predictor() -> CarPricePredictor:
    """
    Return the process-wide CarPricePredictor, loading it on first call.
    The loaded instance is only cached once load_model() succeeds, so a
    failed load never gets stuck as a "poisoned" cache entry — the next
    call will retry the load instead of silently reusing a broken state.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    instance = CarPricePredictor()
    try:
        instance.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise

    _predictor = instance
    return _predictor


def get_known_brands() -> list[str]:
    """
    Returns the exact `brand` categories the loaded pipeline's
    OneHotEncoder was fitted on. Used by:
      - GET /brands, so the frontend can render a dropdown of brands
        the model actually recognizes instead of free text.
      - predict_price(), to flag (not reject) a brand the model has
        never seen — since OneHotEncoder(handle_unknown="ignore")
        would otherwise silently encode it as all-zeros and return a
        confident-looking but essentially uninformed prediction.
    """
    predictor = get_predictor()
    preprocessor = predictor.pipeline.named_steps["preprocessing"]
    encoder = preprocessor.column_transformer_.named_transformers_["cat"].named_steps["onehot"]
    brand_index = preprocessor.categorical_features.index("brand")
    return sorted(encoder.categories_[brand_index].tolist())


def init_predictor() -> None:
    """
    Explicitly load the model at Flask app startup (called from
    app.py). This makes the app FAIL FAST with a clear error if the
    model file is missing, rather than deploying successfully and only
    discovering the problem when the first user hits /predict.
    """
    get_predictor()
    logger.info("Model predictor initialized at app startup.")


def predict_price(payload: CarPredictionRequest) -> dict:
    """
    Runs a validated request through the ML pipeline.

    Parameters
    ----------
    payload : CarPredictionRequest
        Already-validated Pydantic model.

    Returns
    -------
    dict
        {"predicted_price": float, "warning": Optional[str]}
        `warning` is set when `brand` wasn't among the categories the
        model was trained on — the prediction still returns (the
        encoder handles it gracefully), but the caller should know the
        model had no real signal for that specific brand.

    Raises
    ------
    PredictionError
        If the underlying model fails to produce a prediction.
    """
    predictor = get_predictor()
    data = payload.model_dump()
    predicted_price = predictor.predict_one(data)

    warning = None
    known_brands = get_known_brands()
    if data["brand"] not in known_brands:
        warning = (
            f"'{data['brand']}' was not among the brands seen during training. "
            f"This prediction falls back to the model's general pricing pattern "
            f"and may be less accurate than for a recognized brand."
        )
        logger.warning(f"Prediction requested for unrecognized brand '{data['brand']}'.")

    return {"predicted_price": predicted_price, "warning": warning}