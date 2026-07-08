"""
src/predict.py
================
Loads a saved, fully-fitted pipeline (feature engineering +
preprocessing + model, bundled together by train.py/joblib) and
exposes a clean `predict()` interface for new, unseen car data.

This is THE SAME module the Flask API (app/predict.py) wraps — the API
layer only handles HTTP concerns (JSON parsing, status codes); all
actual ML logic lives here, so it's testable independently of Flask
and reusable from a CLI, a notebook, or a batch job.

WHY LOADING ONE PIPELINE FILE IS ENOUGH
--------------------------------------------
Because train.py saved the ENTIRE Pipeline (FeatureEngineer ->
Preprocessor -> Model) as one joblib artifact, this module never needs
to know about medians, encoders, or scalers — calling
`pipeline.predict(raw_dataframe)` runs the whole chain internally,
using the exact statistics learned during training. This is what
guarantees no train/serve skew.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(config.LOG_LEVEL)
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class PredictionError(Exception):
    """Raised when the model fails to load or a prediction fails."""
    pass


class CarPricePredictor:
    """
    Loads a saved model pipeline once and reuses it for multiple
    predictions — avoids the cost of reading the .joblib file from
    disk on every single prediction call (important once this is
    wrapped by a Flask app serving many requests).

    Usage
    -----
    >>> predictor = CarPricePredictor()
    >>> predictor.predict_one({
    ...     "brand": "Maruti", "vehicle_age": 5, "km_driven": 45000,
    ...     "seller_type": "Individual", "fuel_type": "Petrol",
    ...     "transmission_type": "Manual", "mileage": 18.5,
    ...     "engine": 1200, "max_power": 85, "seats": 5
    ... })
    """

    # The exact input fields the model expects, one row's worth. Kept
    # here (mirroring config.NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    # so input validation has a single source of truth.
    REQUIRED_FIELDS = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else config.LATEST_MODEL_FILE
        self.pipeline = None

    def load_model(self) -> None:
        """Load the saved pipeline from disk. Called lazily on first
        prediction if not already loaded, or can be called explicitly
        at app startup (preferred in production — fail fast if the
        model file is missing, rather than on the first user request)."""
        if not self.model_path.exists():
            message = (
                f"No trained model found at '{self.model_path}'. "
                f"Run training and save a model first."
            )
            logger.error(message)
            raise PredictionError(message)

        try:
            logger.info(f"Loading model pipeline from '{self.model_path}'")
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            message = f"Failed to load model: {e}"
            logger.exception(message)
            raise PredictionError(message) from e

    def _validate_input(self, data: dict) -> None:
        """
        Confirms all required fields are present before attempting a
        prediction. This is a structural safety net BEHIND the Flask
        API's Pydantic validation (app/routes.py) — defense in depth,
        so this module is also safe to call directly (e.g. from a
        notebook or test) without going through the API at all.
        """
        missing = [f for f in self.REQUIRED_FIELDS if f not in data]
        if missing:
            message = f"Missing required input fields: {missing}"
            logger.error(message)
            raise PredictionError(message)

    def predict_one(self, data: dict) -> float:
        """
        Predict the selling price for a single car.

        Parameters
        ----------
        data : dict
            Must contain all of CarPricePredictor.REQUIRED_FIELDS,
            e.g. {"brand": "Maruti", "vehicle_age": 5, ...}

        Returns
        -------
        float
            Predicted selling price (in the same unit/scale as the
            training data's target column — lakhs of INR for CarDekho).
        """
        if self.pipeline is None:
            self.load_model()

        try:
            self._validate_input(data)
            input_df = pd.DataFrame([data])

            prediction = self.pipeline.predict(input_df)[0]
            predicted_price = round(float(prediction), 2)

            logger.info(f"Prediction made for input {data}: {predicted_price}")
            return predicted_price

        except PredictionError:
            raise
        except Exception as e:
            message = f"Unexpected error during prediction: {e}"
            logger.exception(message)
            raise PredictionError(message) from e

    def predict_batch(self, records: list) -> list:
        """
        Predict selling prices for multiple cars at once — more
        efficient than calling predict_one() in a loop since the
        pipeline processes the whole batch in one vectorized pass.

        Parameters
        ----------
        records : list[dict]

        Returns
        -------
        list[float]
        """
        if self.pipeline is None:
            self.load_model()

        try:
            for record in records:
                self._validate_input(record)

            input_df = pd.DataFrame(records)
            predictions = self.pipeline.predict(input_df)
            return [round(float(p), 2) for p in predictions]

        except PredictionError:
            raise
        except Exception as e:
            message = f"Unexpected error during batch prediction: {e}"
            logger.exception(message)
            raise PredictionError(message) from e


if __name__ == "__main__":
    # Quick manual sanity check:
    #   python -m src.predict
    sample_input = {
        "brand": "Maruti",
        "vehicle_age": 5,
        "km_driven": 45000,
        "seller_type": "Individual",
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "mileage": 18.5,
        "engine": 1200,
        "max_power": 85,
        "seats": 5,
    }
    try:
        predictor = CarPricePredictor()
        price = predictor.predict_one(sample_input)
        print(f"Predicted selling price: {price}")
    except PredictionError as err:
        logger.error(f"Prediction failed: {err}")
        sys.exit(1)
