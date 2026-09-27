"""
tests/test_api.py
===================
Integration tests for the Flask API endpoints, using Flask's built-in
test client (no real server/network needed).
Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

MODEL_EXISTS = config.LATEST_MODEL_FILE.exists()

pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="No trained model found — run `python -m src.train` and `python -m src.evaluate` first.",
)


@pytest.fixture
def client():
    from app.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def valid_payload():
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


def test_index_returns_200_and_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"ValuGauge" in response.data


def test_api_info_returns_200(client):
    response = client.get("/api")
    assert response.status_code == 200
    assert response.get_json()["status"] == "running"


def test_health_returns_200_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_predict_valid_input_returns_200(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.get_json()
    assert "predicted_price" in body
    assert isinstance(body["predicted_price"], (int, float))


def test_predict_invalid_fuel_type_returns_422(client, valid_payload):
    bad_payload = dict(valid_payload, fuel_type="Rocket Fuel")
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "details" in response.get_json()


def test_predict_negative_km_returns_422(client, valid_payload):
    bad_payload = dict(valid_payload, km_driven=-100)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client, valid_payload):
    incomplete_payload = dict(valid_payload)
    del incomplete_payload["brand"]
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_non_json_body_returns_400(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_predict_wrong_method_returns_405(client):
    response = client.get("/predict")
    assert response.status_code == 405


def test_brands_returns_known_brand_list(client):
    response = client.get("/brands")
    assert response.status_code == 200
    body = response.get_json()
    assert "brands" in body
    assert isinstance(body["brands"], list)
    assert len(body["brands"]) > 0
    # The fixture/production training data always includes "Maruti".
    assert "Maruti" in body["brands"]


def test_predict_known_brand_returns_no_warning(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.get_json()["warning"] is None


def test_predict_unrecognized_brand_returns_warning_not_error(client, valid_payload):
    """
    An unrecognized brand must NOT be rejected outright (the encoder
    handles it gracefully via handle_unknown='ignore') — it should
    still return 200 with a prediction, but flagged with a warning so
    the caller knows the model had no real signal for that brand.
    """
    bad_brand_payload = dict(valid_payload, brand="TotallyMadeUpBrandXYZ")
    response = client.post("/predict", json=bad_brand_payload)
    assert response.status_code == 200
    body = response.get_json()
    assert "predicted_price" in body
    assert body["warning"] is not None
    assert "TotallyMadeUpBrandXYZ" in body["warning"]


def test_predict_rate_limit_returns_429(client, valid_payload):
    """
    Confirms POST /predict is actually rate-limited (config.PREDICT_RATE_LIMIT).
    Sends requests until a 429 is observed, rather than assuming a fixed
    request count, since earlier tests in this module may have already
    consumed part of the current window's quota against the same
    in-memory limiter storage.
    """
    statuses = []
    for _ in range(50):
        response = client.post("/predict", json=valid_payload)
        statuses.append(response.status_code)
        if response.status_code == 429:
            break

    assert 429 in statuses, f"Expected a 429 within 50 requests, got statuses: {statuses}"