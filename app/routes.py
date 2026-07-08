"""
app/routes.py
==============
Defines all HTTP endpoints as a Flask Blueprint.

WHY A BLUEPRINT (rather than routes directly on the Flask app object)
-------------------------------------------------------------------------
Blueprints let routes be defined independently of the app instance,
which is what makes app.py's "application factory" pattern (create_app())
possible. This matters for testing: tests can create a fresh app
instance per test function without route-registration side effects
leaking between tests, and it's the standard structure for any Flask
app expected to grow beyond a handful of routes.
"""

from __future__ import annotations

import logging
# from flask import Blueprint, jsonify, render_template, request

from flask import Blueprint, jsonify, render_template, request
from pydantic import ValidationError

from app.predict import CarPredictionRequest, get_predictor, predict_price
from src.predict import PredictionError

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


@api_bp.route("/", methods=["GET"])
def index():
    """Serves the frontend (templates/index.html)."""
    return render_template("index.html")


@api_bp.route("/api", methods=["GET"])
def api_info():
    """Machine-readable service info, moved off of '/'."""
    return jsonify({
        "service": "Car Price Prediction API",
        "status": "running",
        "endpoints": {
            "GET /": "Frontend web page",
            "GET /api": "This message",
            "GET /health": "Health check (confirms model is loaded)",
            "POST /predict": "Predict a car's selling price from its features",
        },
    }), 200

@api_bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint — used by Docker HEALTHCHECK and Render's
    deployment health monitoring. Confirms not just that Flask is
    running, but that the ML model actually loaded successfully.
    Returns 503 (not 200) if the model isn't ready, so orchestration
    tools correctly treat this instance as unhealthy rather than
    routing traffic to it.
    """
    try:
        get_predictor()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except PredictionError as e:
        logger.error(f"Health check failed — model not loaded: {e}")
        return jsonify({"status": "unhealthy", "model_loaded": False, "error": str(e)}), 503


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predicts a car's selling price from JSON input.

    Expected JSON body:
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

    Responses:
        200 - {"predicted_price": 6.72}
        400 - malformed/missing JSON body
        422 - JSON present but fails schema validation (bad types,
              out-of-range values, invalid category)
        500 - unexpected server/model error
        503 - model not loaded (see /health)
    """
    # ---- 400: no/malformed JSON body ----
    json_data = request.get_json(silent=True)
    if json_data is None:
        logger.warning("POST /predict received no valid JSON body.")
        return jsonify({
            "error": "Request body must be valid JSON with Content-Type: application/json"
        }), 400

    # ---- 422: JSON present but fails schema validation ----
    try:
        payload = CarPredictionRequest(**json_data)
    except ValidationError as e:
        logger.warning(f"POST /predict validation failed: {e.errors()}")
        return jsonify({
            "error": "Invalid input data",
            "details": [
                {"field": ".".join(str(x) for x in err["loc"]), "message": err["msg"]}
                for err in e.errors()
            ],
        }), 422

    # ---- 500: unexpected error during prediction ----
    try:
        predicted_price = predict_price(payload)
        return jsonify({"predicted_price": predicted_price}), 200

    except PredictionError as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    except Exception as e:
        logger.exception(f"Unexpected error in /predict: {e}")
        return jsonify({"error": "Internal server error"}), 500