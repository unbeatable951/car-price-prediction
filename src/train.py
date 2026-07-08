"""
src/train.py
=============
Trains 9 candidate regression models to predict `selling_price`.

KEY ARCHITECTURAL DECISION: FULL PIPELINE PER MODEL
------------------------------------------------------
Each model is wrapped as:

    Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", Preprocessor()),
        ("model", <estimator>),
    ])

This means a SINGLE object encapsulates the ENTIRE flow from raw
dataframe to prediction. Benefits:

  1. NO DATA LEAKAGE: calling `pipeline.fit(X_train, y_train)` fits
     FeatureEngineer's medians/modes AND Preprocessor's scaler/encoder
     using ONLY X_train internally. `cross_val_score` on this pipeline
     correctly refits everything from scratch on each fold — this is
     the textbook-correct way to cross-validate when preprocessing
     involves any learned statistics.

  2. ONE ARTIFACT TO SAVE: `joblib.dump(pipeline, "model.joblib")`
     persists feature engineering + preprocessing + model together.
     At inference time, the Flask API loads ONE file and calls
     `pipeline.predict(raw_dataframe)` — it never needs to know HOW
     the data gets cleaned/encoded, eliminating train/serve skew.

  3. Every model in the comparison sees IDENTICAL preprocessing, so
     Step 6's metric comparison isolates the effect of the algorithm
     itself, not accidental preprocessing differences between models.

WHY THESE 9 MODELS
---------------------
- Linear Regression: simplest possible baseline; if it performs
  competitively, prefer it (lower complexity, more interpretable).
- Ridge / Lasso: linear models with L2/L1 regularization respectively;
  test whether regularization improves on plain linear regression and
  whether Lasso's feature-elimination behavior helps.
- Decision Tree: simplest non-linear model; usually overfits, included
  as a baseline to show WHY ensembles (below) are better.
- Random Forest / Extra Trees: bagging ensembles that reduce the
  Decision Tree's overfitting via averaging many trees.
- Gradient Boosting / XGBoost / CatBoost: boosting ensembles that
  typically achieve the best accuracy on structured/tabular data like
  this — this is why boosted trees dominate Kaggle tabular competitions.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.data_ingestion import DataIngestion, DataIngestionError  # noqa: E402
from src.feature_engineering import FeatureEngineer  # noqa: E402
from src.preprocessing import Preprocessor  # noqa: E402

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(config.LOG_LEVEL)
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class TrainingError(Exception):
    """Raised when data loading, splitting, or model training fails."""
    pass


class ModelTrainer:
    """
    Orchestrates training of multiple candidate regression models,
    each wrapped in an identical feature-engineering + preprocessing
    pipeline for a fair, leakage-free comparison.

    Usage
    -----
    >>> trainer = ModelTrainer()
    >>> X_train, X_test, y_train, y_test = trainer.load_and_split_data()
    >>> fitted_pipelines = trainer.train_all_models(X_train, y_train)
    >>> cv_scores = trainer.cross_validate_all(X_train, y_train)
    """

    def __init__(self, random_state: int = config.RANDOM_STATE):
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Data loading & splitting
    # ------------------------------------------------------------------
    def load_and_split_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads the raw dataset and splits it into train/test BEFORE any
        feature engineering or preprocessing is fit. This ordering is
        deliberate: fitting FeatureEngineer/Preprocessor on the full
        dataset (including test rows) would leak test-set statistics
        (medians, encoder categories) into training — this split-first
        order prevents that entirely.

        Returns
        -------
        (X_train, X_test, y_train, y_test)
            X_train/X_test are still RAW (uncleaned) dataframes — the
            pipeline's FeatureEngineer step cleans them internally
            during .fit()/.predict().
        """
        try:
            raw_df = DataIngestion().load_data()

            if config.TARGET_COLUMN not in raw_df.columns:
                message = f"Target column '{config.TARGET_COLUMN}' missing from raw data."
                logger.error(message)
                raise TrainingError(message)

            X = raw_df.drop(columns=[config.TARGET_COLUMN])
            y = raw_df[config.TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=self.random_state
            )

            logger.info(
                f"Data split complete — Train: {X_train.shape}, Test: {X_test.shape}"
            )
            return X_train, X_test, y_train, y_test

        except DataIngestionError as e:
            raise TrainingError(f"Training aborted — data ingestion failed: {e}") from e
        except Exception as e:
            message = f"Unexpected error during data loading/splitting: {e}"
            logger.exception(message)
            raise TrainingError(message) from e

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    def get_model_candidates(self) -> dict:
        """
        Returns the 9 candidate models. Centralized here (rather than
        scattered inline) so adding/removing a candidate model is a
        one-line change.
        """
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=self.random_state),
            "Lasso": Lasso(random_state=self.random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
            "Random Forest": RandomForestRegressor(
                random_state=self.random_state, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesRegressor(
                random_state=self.random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=self.random_state
            ),
            "XGBoost": XGBRegressor(
                random_state=self.random_state, n_jobs=-1, verbosity=0
            ),
            "CatBoost": CatBoostRegressor(
                random_state=self.random_state, verbose=0
            ),
        }

    def build_pipeline(self, model) -> Pipeline:
        """
        Wraps a single estimator with the shared feature-engineering
        and preprocessing steps. A NEW FeatureEngineer/Preprocessor
        instance is created per pipeline (not shared/reused) so each
        model's pipeline is fully independent — fitting one pipeline
        can never accidentally mutate another's learned statistics.
        """
        return Pipeline(steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessing", Preprocessor()),
            ("model", model),
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_all_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> dict:
        """
        Fits one full pipeline per candidate model on the training data.

        Returns
        -------
        dict[str, dict]
            Mapping of model name -> {"pipeline": fitted Pipeline,
            "train_time_seconds": float}
        """
        fitted = {}

        for name, model in self.get_model_candidates().items():
            try:
                logger.info(f"Training '{name}'...")
                pipeline = self.build_pipeline(model)

                start = time.time()
                pipeline.fit(X_train, y_train)
                elapsed = round(time.time() - start, 3)

                fitted[name] = {"pipeline": pipeline, "train_time_seconds": elapsed}
                logger.info(f"'{name}' trained in {elapsed}s")

            except Exception as e:
                # A single model failing (e.g. a library-specific quirk)
                # should NOT abort training of the other 8 models.
                logger.error(f"Training failed for '{name}': {e}")
                continue

        if not fitted:
            raise TrainingError("All candidate models failed to train.")

        return fitted

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    def cross_validate_all(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = config.CV_FOLDS
    ) -> dict:
        """
        Runs K-Fold cross-validation for every candidate model, refitting
        the FULL pipeline (feature engineering + preprocessing + model)
        on each fold. This is deliberately more expensive than
        cross-validating just the model, but it's the only way to get
        an honest estimate of real-world performance when preprocessing
        itself involves learned statistics.

        Returns
        -------
        dict[str, dict]
            Mapping of model name -> {"cv_mean_r2": float, "cv_std_r2": float}
        """
        results = {}
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for name, model in self.get_model_candidates().items():
            try:
                logger.info(f"Cross-validating '{name}' ({cv}-fold)...")
                pipeline = self.build_pipeline(model)

                scores = cross_val_score(
                    pipeline, X_train, y_train, cv=kfold, scoring="r2", n_jobs=1
                )
                results[name] = {
                    "cv_mean_r2": round(scores.mean(), 4),
                    "cv_std_r2": round(scores.std(), 4),
                }
                logger.info(
                    f"'{name}' CV R2: {results[name]['cv_mean_r2']} "
                    f"(+/- {results[name]['cv_std_r2']})"
                )

            except Exception as e:
                logger.error(f"Cross-validation failed for '{name}': {e}")
                continue

        return results


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.load_and_split_data()

        fitted_pipelines = trainer.train_all_models(X_train, y_train)
        cv_results = trainer.cross_validate_all(X_train, y_train)

        # Persist everything evaluate.py needs, so evaluation doesn't
        # require retraining from scratch.
        artifact_bundle = {
            "fitted_pipelines": fitted_pipelines,
            "cv_results": cv_results,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        bundle_path = config.ARTIFACTS_DIR / "training_bundle.joblib"
        joblib.dump(artifact_bundle, bundle_path)
        logger.info(f"Training bundle saved to '{bundle_path}'")

        print("\nTraining complete. Models trained:")
        for name, info in fitted_pipelines.items():
            print(f"  - {name} ({info['train_time_seconds']}s)")

    except TrainingError as err:
        logger.error(f"Training pipeline aborted: {err}")
        sys.exit(1)
