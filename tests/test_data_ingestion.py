"""
tests/test_data_ingestion.py
==============================
Unit tests for src/data_ingestion.py, using pytest's tmp_path fixture
to write small synthetic CSVs — no dependency on the real CarDekho
dataset (data/cardekho_dataset.csv, which isn't committed to the repo;
see data/README.md), so these tests always run, everywhere.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_ingestion import DataIngestion, DataIngestionError, load_raw_data


@pytest.fixture
def valid_csv(tmp_path):
    """A minimal CSV with the target column and a couple of features."""
    csv_path = tmp_path / "valid_dataset.csv"
    pd.DataFrame(
        {
            "brand": ["Maruti", "Hyundai", "Honda"],
            "vehicle_age": [5, 3, 8],
            "km_driven": [45000, 20000, 90000],
            "selling_price": [5.5, 8.2, 3.1],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def test_load_data_returns_dataframe(valid_csv):
    df = DataIngestion(file_path=valid_csv).load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert config.TARGET_COLUMN in df.columns


def test_load_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(DataIngestionError, match="not found"):
        DataIngestion(file_path=missing_path).load_data()


def test_load_data_empty_csv_raises(tmp_path):
    empty_path = tmp_path / "empty.csv"
    # Header row only, no data rows.
    pd.DataFrame(columns=["brand", "vehicle_age", "selling_price"]).to_csv(empty_path, index=False)
    with pytest.raises(DataIngestionError, match="empty"):
        DataIngestion(file_path=empty_path).load_data()


def test_load_data_missing_target_column_raises(tmp_path):
    bad_path = tmp_path / "no_target.csv"
    pd.DataFrame({"brand": ["Maruti", "Hyundai"], "vehicle_age": [5, 3]}).to_csv(bad_path, index=False)
    with pytest.raises(DataIngestionError, match=config.TARGET_COLUMN):
        DataIngestion(file_path=bad_path).load_data()


def test_load_data_target_column_case_insensitive(tmp_path):
    """
    _validate_schema() normalizes column names (strip + lowercase)
    before checking for the target column, since real-world CSV
    exports vary in casing (e.g. 'Selling_Price' vs 'selling_price').
    """
    mixed_case_path = tmp_path / "mixed_case.csv"
    pd.DataFrame(
        {
            "Brand": ["Maruti"],
            "Vehicle_Age": [5],
            config.TARGET_COLUMN.upper(): [5.5],
        }
    ).to_csv(mixed_case_path, index=False)
    df = DataIngestion(file_path=mixed_case_path).load_data()
    assert len(df) == 1


def test_get_basic_info_before_load_raises():
    ingestion = DataIngestion(file_path=Path("irrelevant.csv"))
    with pytest.raises(DataIngestionError, match="No data loaded"):
        ingestion.get_basic_info()


def test_get_basic_info_after_load_returns_summary(valid_csv):
    ingestion = DataIngestion(file_path=valid_csv)
    ingestion.load_data()
    info = ingestion.get_basic_info()

    assert info["num_rows"] == 3
    assert info["num_columns"] == 4
    assert set(info["columns"]) == {"brand", "vehicle_age", "km_driven", "selling_price"}
    assert info["duplicate_rows"] == 0


def test_load_raw_data_convenience_function(valid_csv):
    df = load_raw_data(file_path=valid_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3