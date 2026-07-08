"""
src/preprocessing.py
======================
Second stage of the ML pipeline: encodes categorical features and
scales numeric features, producing the final numeric matrix that
models are trained on.

WHY THIS IS SEPARATE FROM feature_engineering.py
---------------------------------------------------
`feature_engineering.py` handles things that are safe to define BEFORE
any train/test split consideration (dropping known-junk columns) or
that must be learned strictly on training data (imputation medians).

`preprocessing.py` handles ENCODING and SCALING, which have a stricter
requirement: the encoder/scaler must be `fit()` on the TRAINING split
only, then applied (`transform()`) unchanged to validation, test, and
every future live prediction. Keeping this as its own composable stage
makes that boundary explicit and enforced by the code structure itself,
not just a comment.

WHY ONE-HOT ENCODING FOR CATEGORICALS
----------------------------------------
`brand`, `seller_type`, `fuel_type`, `transmission_type` are NOMINAL
categories — there's no inherent order (e.g. "Diesel" isn't
mathematically "more" than "Petrol"). One-hot encoding avoids the
false ordinal relationship that label encoding (0,1,2,3...) would
otherwise impose, which would mislead linear models in particular.

`handle_unknown="ignore"` is critical for production: if a car brand
that never appeared in training data (e.g. a brand new EV brand) shows
up in a live prediction request, the encoder should represent it as
all-zeros for that feature rather than crashing the API.

WHY STANDARD SCALING FOR NUMERICS
-------------------------------------
`km_driven` ranges in the tens of thousands, while `seats` ranges
2-10 and `mileage` ranges roughly 10-30. Without scaling, distance-
and gradient-based algorithms (Linear/Ridge/Lasso Regression) would
let large-magnitude features dominate purely due to scale, not actual
importance. Tree-based models (Random Forest, XGBoost, CatBoost,
Decision Tree) are scale-invariant and don't strictly need this — but
using ONE consistent preprocessed feature matrix for ALL models keeps
the comparison in Step 6 (model evaluation) fair and the codebase
simple. This tradeoff is deliberate and explained again in train.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


class PreprocessingError(Exception):
    """Raised when encoding/scaling fails or is used incorrectly."""
    pass


def build_preprocessor(
    numerical_features: Optional[list] = None,
    categorical_features: Optional[list] = None,
) -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer that applies:
      - StandardScaler to numeric columns
      - OneHotEncoder to categorical columns

    Returns
    -------
    ColumnTransformer
        Unfit transformer — call .fit() or .fit_transform() on training
        data before using it to .transform() anything else.
    """
    numerical_features = numerical_features or config.NUMERICAL_FEATURES
    categorical_features = categorical_features or config.CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",  # any column not explicitly listed is dropped,
                           # guaranteeing the output matrix shape is
                           # always predictable regardless of stray
                           # columns in the input.
    )

    logger.info(
        f"Built ColumnTransformer — numeric: {numerical_features}, "
        f"categorical: {categorical_features}"
    )
    return preprocessor


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Wraps the sklearn ColumnTransformer with:
      - Clear fit/transform contract matching FeatureEngineer's style
      - Human-readable output feature names (for feature importance
        plots later, since one-hot encoding otherwise produces opaque
        names like 'cat__x0_Petrol')
      - Save/load via joblib so training and inference use the exact
        same fitted encoder/scaler.
    """

    def __init__(
        self,
        numerical_features: Optional[list] = None,
        categorical_features: Optional[list] = None,
    ):
        self.numerical_features = numerical_features or config.NUMERICAL_FEATURES
        self.categorical_features = categorical_features or config.CATEGORICAL_FEATURES
        self.column_transformer_: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y=None) -> "Preprocessor":
        try:
            missing = set(self.numerical_features + self.categorical_features) - set(X.columns)
            if missing:
                message = f"Preprocessor.fit(): missing expected columns {sorted(missing)}"
                logger.error(message)
                raise PreprocessingError(message)

            self.column_transformer_ = build_preprocessor(
                self.numerical_features, self.categorical_features
            )
            self.column_transformer_.fit(X)

            logger.info(
                f"Preprocessor fitted. Output feature count: "
                f"{len(self.get_feature_names_out())}"
            )
            return self

        except PreprocessingError:
            raise
        except Exception as e:
            message = f"Unexpected error during Preprocessor.fit(): {e}"
            logger.exception(message)
            raise PreprocessingError(message) from e

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.column_transformer_ is None:
            message = "Preprocessor.transform() called before fit(). Call fit() first."
            logger.error(message)
            raise PreprocessingError(message)

        try:
            return self.column_transformer_.transform(X)
        except Exception as e:
            message = f"Unexpected error during Preprocessor.transform(): {e}"
            logger.exception(message)
            raise PreprocessingError(message) from e

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list:
        """
        Returns human-readable output column names, e.g.
        ['vehicle_age', 'km_driven', ..., 'brand_Maruti', 'brand_Honda',
        'fuel_type_Petrol', ...]. Needed later for feature importance
        plots — otherwise sklearn's default names ('num__vehicle_age',
        'cat__onehot__x3_Petrol') are unreadable in a chart.
        """
        if self.column_transformer_ is None:
            raise PreprocessingError("Cannot get feature names before fit().")

        num_names = self.numerical_features
        cat_encoder = self.column_transformer_.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        return list(num_names) + cat_names

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted preprocessor to disk with joblib."""
        if self.column_transformer_ is None:
            raise PreprocessingError("Cannot save an unfitted Preprocessor.")

        save_path = Path(path) if path else config.PREPROCESSOR_FILE
        joblib.dump(self, save_path)
        logger.info(f"Preprocessor saved to '{save_path}'")
        return save_path

    @staticmethod
    def load(path: Optional[Path] = None) -> "Preprocessor":
        """Load a previously fitted Preprocessor from disk."""
        load_path = Path(path) if path else config.PREPROCESSOR_FILE
        if not load_path.exists():
            message = f"No saved preprocessor found at '{load_path}'"
            logger.error(message)
            raise PreprocessingError(message)

        logger.info(f"Loading preprocessor from '{load_path}'")
        return joblib.load(load_path)


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.feature_engineering import engineer_features

    try:
        raw_df = DataIngestion().load_data()
        cleaned_df, fitted_engineer = engineer_features(raw_df)

        X = cleaned_df.drop(columns=[config.TARGET_COLUMN])
        y = cleaned_df[config.TARGET_COLUMN]

        preprocessor = Preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        print(f"Input shape: {X.shape} -> Output shape: {X_transformed.shape}")
        print(f"Output feature names ({len(preprocessor.get_feature_names_out())}):")
        print(preprocessor.get_feature_names_out())

        preprocessor.save()
    except Exception as err:
        logger.error(f"Preprocessing failed: {err}")
        sys.exit(1)
