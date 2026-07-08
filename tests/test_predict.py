"""
tests/test_predict.py
=======================
Unit tests for CarPricePredictor (src/predict.py).
Requires a trained model to already exist at config.LATEST_MODEL_FILE
(run `python -m src.train` then `python -m src.evaluate` first).
Run with: pytest tests/test_predict.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.predict import CarPricePredictor, PredictionError

MODEL_EXISTS = config.LATEST_MODEL_FILE.exists()

pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="No trained model found — run `python -m src.train` and `python -m src.evaluate` first.",
)


@pytest.fixture
def valid_input():
    return {
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


def test_predict_one_returns_float(valid_input):
    predictor = CarPricePredictor()
    price = predictor.predict_one(valid_input)
    assert isinstance(price, float)
    assert price >= 0


def test_predict_missing_field_raises(valid_input):
    predictor = CarPricePredictor()
    incomplete_input = dict(valid_input)
    del incomplete_input["brand"]
    with pytest.raises(PredictionError):
        predictor.predict_one(incomplete_input)


def test_predict_batch_returns_list(valid_input):
    predictor = CarPricePredictor()
    prices = predictor.predict_batch([valid_input, valid_input])
    assert len(prices) == 2
    assert all(isinstance(p, float) for p in prices)


def test_missing_model_file_raises(tmp_path):
    predictor = CarPricePredictor(model_path=tmp_path / "nonexistent.joblib")
    with pytest.raises(PredictionError):
        predictor.load_model()
