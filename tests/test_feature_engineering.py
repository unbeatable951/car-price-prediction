"""
tests/test_feature_engineering.py
====================================
Unit tests for the FeatureEngineer custom transformer.
Run with: pytest tests/test_feature_engineering.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import FeatureEngineer, FeatureEngineeringError, engineer_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "car_name": ["Maruti Alto", "Honda City", "Toyota Innova"],
        "brand": ["Maruti", "Honda", np.nan],
        "model": ["Alto", "City", "Innova"],
        "vehicle_age": [5, np.nan, 2],
        "km_driven": [50000, 30000, np.nan],
        "seller_type": ["Individual", "Dealer", "Individual"],
        "fuel_type": ["Petrol", "Diesel", "Petrol"],
        "transmission_type": ["Manual", "Automatic", "Automatic"],
        "mileage": [18.5, 17.0, 15.0],
        "engine": [1000, 1500, 2500],
        "max_power": [65, 100, 150],
        "seats": [5, 5, 7],
        "selling_price": [3.5, 8.1, 15.0],
    })


def test_drops_unnecessary_columns(sample_df):
    cleaned, _ = engineer_features(sample_df)
    for col in ["Unnamed: 0", "car_name", "model"]:
        assert col not in cleaned.columns


def test_imputes_missing_numeric_values(sample_df):
    cleaned, _ = engineer_features(sample_df)
    assert cleaned["vehicle_age"].isnull().sum() == 0
    assert cleaned["km_driven"].isnull().sum() == 0


def test_imputes_missing_categorical_values(sample_df):
    cleaned, _ = engineer_features(sample_df)
    assert cleaned["brand"].isnull().sum() == 0


def test_transform_before_fit_raises():
    fe = FeatureEngineer()
    with pytest.raises(FeatureEngineeringError):
        fe.transform(pd.DataFrame({"vehicle_age": [1]}))


def test_inference_uses_training_statistics(sample_df):
    """A single-row inference request should be imputed using stats
    learned from training data, not recomputed from itself."""
    _, fitted_engineer = engineer_features(sample_df)

    new_row = pd.DataFrame({
        "brand": ["Maruti"], "vehicle_age": [np.nan], "km_driven": [10000],
        "seller_type": ["Dealer"], "fuel_type": ["Petrol"],
        "transmission_type": ["Manual"], "mileage": [18.0],
        "engine": [1000], "max_power": [65], "seats": [5],
    })
    result, _ = engineer_features(new_row, fitted_engineer=fitted_engineer)
    assert result["vehicle_age"].iloc[0] == fitted_engineer.numeric_medians_["vehicle_age"]


def test_missing_required_columns_raises():
    fe = FeatureEngineer()
    incomplete_df = pd.DataFrame({"brand": ["Maruti"]})  # missing most required columns
    with pytest.raises(FeatureEngineeringError):
        fe.fit(incomplete_df)
