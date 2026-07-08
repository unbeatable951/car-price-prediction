"""
src/feature_engineering.py
============================
Transforms the RAW dataframe (as loaded by data_ingestion.py) into a
clean, model-ready dataframe: dropping unusable columns, validating the
schema, and handling missing values.

WHY A SKLEARN-COMPATIBLE CUSTOM TRANSFORMER?
----------------------------------------------
Instead of a bag of loose functions, `FeatureEngineer` inherits from
`sklearn.base.BaseEstimator` and `TransformerMixin`. This gives us:

  1. Drop-in compatibility with sklearn Pipelines:
         Pipeline([
             ("feature_engineering", FeatureEngineer()),
             ("preprocessing", preprocessor),   # from preprocessing.py
             ("model", RandomForestRegressor()),
         ])
     The ENTIRE flow — raw dataframe in, prediction out — becomes one
     object that can be saved with joblib and reused identically in
     training, evaluation, and the live Flask API. This eliminates the
     classic bug where training-time and inference-time preprocessing
     silently drift apart.

  2. `fit()` / `transform()` separation: `fit()` learns anything that
     must be learned from training data ONLY (e.g. median values used
     to impute missing numbers), and `transform()` applies it. This
     prevents data leakage from test data into training decisions.

  3. Free compatibility with GridSearchCV, cross_val_score, etc. since
     sklearn "just works" with any class implementing this interface.

WHY THIS DATASET NEEDS LESS ENGINEERING THAN A TYPICAL CARDEKHO PROJECT
-------------------------------------------------------------------------
Most CarDekho tutorials show "convert Year -> Vehicle_Age". Our actual
file (data/cardekho_dataset.csv) already ships `vehicle_age` pre-computed,
so that conversion is unnecessary here — doing it anyway would just
recreate a column that already exists. What DOES still need engineering:
  - Dropping unusable columns (index leftover, high-cardinality strings)
  - Validating required columns exist before proceeding
  - Handling missing values in numeric/categorical columns
Categorical ENCODING and numeric SCALING are handled separately in
`preprocessing.py` — kept in a different file because encoding/scaling
must be *fit* only on the training split (to avoid leakage), while
column-dropping and missing-value logic here are safe to define once,
independent of the split.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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


class FeatureEngineeringError(Exception):
    """Raised when the dataframe fails schema validation or transformation."""
    pass


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible transformer that cleans the raw car
    dataframe into a model-ready shape.

    Steps performed (in order):
        1. Validate that all required raw columns are present.
        2. Drop unnecessary columns (config.COLUMNS_TO_DROP).
        3. Impute missing numeric values with the TRAINING median.
        4. Impute missing categorical values with the TRAINING mode.

    Parameters
    ----------
    columns_to_drop : list[str], optional
        Columns to remove. Defaults to config.COLUMNS_TO_DROP.

    Attributes learned during fit()
    --------------------------------
    numeric_medians_ : dict
        Median of each numeric column, learned ONLY from training data.
    categorical_modes_ : dict
        Mode (most frequent value) of each categorical column, learned
        ONLY from training data.
    """

    def __init__(self, columns_to_drop: Optional[list] = None):
        self.columns_to_drop = columns_to_drop or config.COLUMNS_TO_DROP

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Ensure every column we expect to either use or drop is actually
        present, EXCEPT the target column (which won't exist at
        inference time for a single prediction request).
        """
        expected = set(config.NUMERICAL_FEATURES) | set(config.CATEGORICAL_FEATURES)
        available = set(df.columns)
        missing = expected - available

        if missing:
            message = (
                f"FeatureEngineer expected columns {sorted(missing)} but "
                f"they were not found. Available columns: {list(df.columns)}"
            )
            logger.error(message)
            raise FeatureEngineeringError(message)

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop configured unnecessary columns. Uses errors='ignore' so
        this doesn't crash if a column is already absent (e.g. at
        inference time, where the incoming payload never included
        `car_name` or `Unnamed: 0` in the first place).
        """
        existing_to_drop = [c for c in self.columns_to_drop if c in df.columns]
        if existing_to_drop:
            logger.info(f"Dropping columns: {existing_to_drop}")
        return df.drop(columns=existing_to_drop, errors="ignore")

    # ------------------------------------------------------------------
    # sklearn-required interface
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """
        Learn imputation statistics from the TRAINING data only.

        CRITICAL: this must only ever be called on the training split.
        Calling it on the full dataset (before train/test split) would
        leak test-set information into the median/mode values used to
        fill missing training rows — a subtle but real form of data
        leakage that inflates validation metrics.
        """
        try:
            df = X.copy()
            self._validate_required_columns(df)
            df = self._drop_columns(df)

            self.numeric_medians_ = {
                col: df[col].median()
                for col in config.NUMERICAL_FEATURES
                if col in df.columns
            }
            self.categorical_modes_ = {
                col: df[col].mode(dropna=True).iloc[0]
                for col in config.CATEGORICAL_FEATURES
                if col in df.columns and not df[col].mode(dropna=True).empty
            }

            logger.info(f"FeatureEngineer fitted. Numeric medians: {self.numeric_medians_}")
            logger.info(f"FeatureEngineer fitted. Categorical modes: {self.categorical_modes_}")
            return self

        except FeatureEngineeringError:
            raise
        except Exception as e:
            message = f"Unexpected error during FeatureEngineer.fit(): {e}"
            logger.exception(message)
            raise FeatureEngineeringError(message) from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column dropping and missing-value imputation using
        statistics learned in fit(). Safe to call on train, validation,
        test, OR a single-row inference payload.
        """
        try:
            if not hasattr(self, "numeric_medians_"):
                message = "FeatureEngineer.transform() called before fit(). Call fit() first."
                logger.error(message)
                raise FeatureEngineeringError(message)

            df = X.copy()
            df = self._drop_columns(df)

            # Impute numeric columns with TRAINING medians (not the
            # current batch's own median — that would be inconsistent
            # between training and a single live prediction request).
            for col, median_value in self.numeric_medians_.items():
                if col in df.columns and df[col].isnull().any():
                    n_missing = df[col].isnull().sum()
                    logger.info(f"Imputing {n_missing} missing values in '{col}' with median={median_value}")
                    df[col] = df[col].fillna(median_value)

            # Impute categorical columns with TRAINING mode.
            for col, mode_value in self.categorical_modes_.items():
                if col in df.columns and df[col].isnull().any():
                    n_missing = df[col].isnull().sum()
                    logger.info(f"Imputing {n_missing} missing values in '{col}' with mode='{mode_value}'")
                    df[col] = df[col].fillna(mode_value)

            # Normalize categorical text (strip whitespace, consistent
            # casing) so "Petrol" and "petrol " don't become two
            # different one-hot categories downstream.
            for col in config.CATEGORICAL_FEATURES:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

            return df

        except FeatureEngineeringError:
            raise
        except Exception as e:
            message = f"Unexpected error during FeatureEngineer.transform(): {e}"
            logger.exception(message)
            raise FeatureEngineeringError(message) from e

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Explicit override (TransformerMixin provides this by default,
        but defining it directly makes the fit-then-transform intent
        obvious to anyone reading this class)."""
        return self.fit(X, y).transform(X)


def engineer_features(
    df: pd.DataFrame, fitted_engineer: Optional[FeatureEngineer] = None
) -> tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience function for scripts/notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe to clean.
    fitted_engineer : FeatureEngineer, optional
        If provided (e.g. loaded from a saved training pipeline), reuse
        its learned statistics instead of fitting new ones. Use this
        at INFERENCE time. Leave None to fit fresh — use this only on
        TRAINING data.

    Returns
    -------
    (cleaned_df, fitted_engineer)
    """
    engineer = fitted_engineer or FeatureEngineer()
    if fitted_engineer is None:
        cleaned = engineer.fit_transform(df)
    else:
        cleaned = engineer.transform(df)
    return cleaned, engineer


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion

    try:
        raw_df = DataIngestion().load_data()
        cleaned_df, fitted = engineer_features(raw_df)
        print(f"Before: {raw_df.shape} -> After: {cleaned_df.shape}")
        print(cleaned_df.head())
    except Exception as err:
        logger.error(f"Feature engineering failed: {err}")
        sys.exit(1)
