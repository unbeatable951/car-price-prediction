"""
tests/test_preprocessing.py
=============================
Unit tests for the Preprocessor (encoding + scaling) transformer.
Run with: pytest tests/test_preprocessing.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import Preprocessor, PreprocessingError


@pytest.fixture
def sample_clean_df():
    return pd.DataFrame({
        "vehicle_age": [5, 3, 7, 2],
        "km_driven": [50000, 30000, 60000, 20000],
        "mileage": [18.5, 20.0, 17.0, 19.0],
        "engine": [1000, 1200, 1500, 1200],
        "max_power": [65, 85, 100, 85],
        "seats": [5, 5, 5, 7],
        "brand": ["Maruti", "Hyundai", "Honda", "Maruti"],
        "seller_type": ["Individual", "Dealer", "Individual", "Dealer"],
        "fuel_type": ["Petrol", "Diesel", "Petrol", "Petrol"],
        "transmission_type": ["Manual", "Manual", "Automatic", "Manual"],
    })


def test_fit_transform_produces_numeric_matrix(sample_clean_df):
    pre = Preprocessor()
    X = pre.fit_transform(sample_clean_df)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 4
    assert np.issubdtype(X.dtype, np.number)


def test_transform_before_fit_raises(sample_clean_df):
    pre = Preprocessor()
    with pytest.raises(PreprocessingError):
        pre.transform(sample_clean_df)


def test_feature_names_are_human_readable(sample_clean_df):
    pre = Preprocessor()
    pre.fit(sample_clean_df)
    names = pre.get_feature_names_out()
    assert "vehicle_age" in names
    assert any("brand_" in n for n in names)


def test_unseen_category_does_not_crash(sample_clean_df):
    pre = Preprocessor()
    pre.fit(sample_clean_df)

    new_row = sample_clean_df.iloc[[0]].copy()
    new_row["brand"] = "BrandNewEV"  # never seen during fit
    result = pre.transform(new_row)
    assert result.shape[0] == 1  # doesn't crash, still returns one row


def test_save_and_load_roundtrip(sample_clean_df, tmp_path):
    pre = Preprocessor()
    X_original = pre.fit_transform(sample_clean_df)

    save_path = tmp_path / "test_preprocessor.joblib"
    pre.save(save_path)

    loaded = Preprocessor.load(save_path)
    X_loaded = loaded.transform(sample_clean_df)

    assert np.allclose(X_original, X_loaded)
