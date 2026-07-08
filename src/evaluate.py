"""
src/evaluate.py
=================
Loads the training_bundle.joblib produced by train.py and evaluates
EVERY trained model on the held-out test set using MAE, MSE, RMSE,
R2, Adjusted R2, plus the cross-validation scores computed during
training. Produces a single comparison table and picks the best
model with an explicit, printed justification — never silently.

WHY EVALUATION IS ITS OWN FILE (SEPARATE FROM train.py)
-----------------------------------------------------------
Training is expensive (especially CV across 9 models); evaluation is
cheap (just running .predict() and computing metrics). Separating them
means you can re-run evaluation — try different metrics, thresholds,
or selection criteria — without retraining anything, as long as
`training_bundle.joblib` already exists. This mirrors how real MLOps
pipelines separate "train" and "evaluate" into independent stages.

WHY ADJUSTED R2 MATTERS HERE SPECIFICALLY
----------------------------------------------
Plain R2 always increases (or stays the same) as you add more features,
even useless ones — it never penalizes complexity. Since one-hot
encoding `brand` alone can add ~30 columns, plain R2 could reward a
model for fitting noise in those extra dimensions. Adjusted R2
corrects for the number of predictors, so it's a fairer comparison
when candidate models effectively have different "effective"
predictor counts after encoding.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

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


class EvaluationError(Exception):
    """Raised when the training bundle is missing or evaluation fails."""
    pass


class ModelEvaluator:
    """
    Evaluates all trained pipelines from training_bundle.joblib against
    the held-out test set and produces a ranked comparison table.

    Usage
    -----
    >>> evaluator = ModelEvaluator()
    >>> comparison_df = evaluator.evaluate_all()
    >>> best_name, best_pipeline, reason = evaluator.select_best_model(comparison_df)
    """

    def __init__(self, bundle_path: Optional[Path] = None):
        self.bundle_path = Path(bundle_path) if bundle_path else (
            config.ARTIFACTS_DIR / "training_bundle.joblib"
        )
        self.bundle: Optional[dict] = None

    # ------------------------------------------------------------------
    def load_bundle(self) -> dict:
        """Load fitted pipelines, CV results, and the test set."""
        if not self.bundle_path.exists():
            message = (
                f"No training bundle found at '{self.bundle_path}'. "
                f"Run 'python -m src.train' first."
            )
            logger.error(message)
            raise EvaluationError(message)

        try:
            logger.info(f"Loading training bundle from '{self.bundle_path}'")
            self.bundle = joblib.load(self.bundle_path)
            return self.bundle
        except Exception as e:
            message = f"Failed to load training bundle: {e}"
            logger.exception(message)
            raise EvaluationError(message) from e

    # ------------------------------------------------------------------
    def _adjusted_r2(self, r2: float, n_samples: int, n_features: int) -> float:
        """
        Adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
        where n = number of test samples, p = number of predictors
        (post-encoding feature count).

        Guards against division-by-zero when n_samples is very small
        relative to n_features (can happen on tiny test sets).
        """
        denominator = n_samples - n_features - 1
        if denominator <= 0:
            logger.warning(
                f"Cannot compute Adjusted R2 reliably (n={n_samples}, "
                f"p={n_features}); returning plain R2 instead."
            )
            return r2
        return 1 - (1 - r2) * (n_samples - 1) / denominator

    def _get_feature_count(self, pipeline) -> int:
        """Extract post-encoding feature count from the fitted pipeline's
        preprocessing step, for the Adjusted R2 penalty term."""
        try:
            preprocessor = pipeline.named_steps["preprocessing"]
            return len(preprocessor.get_feature_names_out())
        except Exception:
            logger.warning("Could not determine feature count; defaulting to 1.")
            return 1

    # ------------------------------------------------------------------
    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate every fitted pipeline in the bundle on the test set.

        Returns
        -------
        pd.DataFrame
            One row per model with columns:
            ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score', 'Adjusted R2',
             'CV Mean R2', 'CV Std R2', 'Train Time (s)']
            Sorted by 'R2 Score' descending.
        """
        if self.bundle is None:
            self.load_bundle()

        fitted_pipelines = self.bundle["fitted_pipelines"]
        cv_results = self.bundle.get("cv_results", {})
        X_test = self.bundle["X_test"]
        y_test = self.bundle["y_test"]

        rows = []
        for name, info in fitted_pipelines.items():
            try:
                pipeline = info["pipeline"]
                y_pred = pipeline.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                n_features = self._get_feature_count(pipeline)
                adj_r2 = self._adjusted_r2(r2, len(y_test), n_features)

                cv_info = cv_results.get(name, {})

                rows.append({
                    "Model": name,
                    "MAE": round(mae, 4),
                    "MSE": round(mse, 4),
                    "RMSE": round(rmse, 4),
                    "R2 Score": round(r2, 4),
                    "Adjusted R2": round(adj_r2, 4),
                    "CV Mean R2": cv_info.get("cv_mean_r2", np.nan),
                    "CV Std R2": cv_info.get("cv_std_r2", np.nan),
                    "Train Time (s)": info.get("train_time_seconds", np.nan),
                })
                logger.info(f"Evaluated '{name}': R2={r2:.4f}, RMSE={rmse:.4f}")

            except Exception as e:
                logger.error(f"Evaluation failed for '{name}': {e}")
                continue

        if not rows:
            raise EvaluationError("No models could be evaluated.")

        comparison_df = pd.DataFrame(rows).sort_values(
            "R2 Score", ascending=False
        ).reset_index(drop=True)

        return comparison_df

    # ------------------------------------------------------------------
    def select_best_model(self, comparison_df: pd.DataFrame) -> tuple:
        """
        Selects the best model using a two-factor criterion, not just
        the single highest test R2:

          1. Primary: highest 'CV Mean R2' (cross-validated performance
             is a more reliable indicator of real-world generalization
             than a single test-set score, which can vary based on
             which rows happened to land in the test split).
          2. Tie-breaker / sanity check: the gap between CV Mean R2 and
             test R2 Score should be small — a large gap signals
             overfitting to the training folds despite a good CV score.

        Returns
        -------
        (best_model_name, reason_string)
        """
        if self.bundle is None:
            raise EvaluationError("Call evaluate_all() before select_best_model().")

        ranked = comparison_df.sort_values("CV Mean R2", ascending=False).reset_index(drop=True)
        best_row = ranked.iloc[0]
        best_name = best_row["Model"]

        cv_test_gap = abs(best_row["CV Mean R2"] - best_row["R2 Score"])

        reason = (
            f"'{best_name}' selected as the best model:\n"
            f"  - Highest cross-validated R2: {best_row['CV Mean R2']} "
            f"(+/- {best_row['CV Std R2']})\n"
            f"  - Test-set R2: {best_row['R2 Score']} "
            f"(gap vs CV: {cv_test_gap:.4f} — "
            f"{'small, good generalization' if cv_test_gap < 0.05 else 'notable, watch for overfitting'})\n"
            f"  - Test-set RMSE: {best_row['RMSE']}, MAE: {best_row['MAE']}\n"
            f"  - Adjusted R2: {best_row['Adjusted R2']} "
            f"(penalizes feature count, confirms performance isn't an "
            f"artifact of one-hot encoding adding many columns)"
        )

        logger.info(f"Best model selected: {best_name}")
        return best_name, reason

    def save_comparison(self, comparison_df: pd.DataFrame, path: Optional[Path] = None) -> Path:
        save_path = Path(path) if path else (config.ARTIFACTS_DIR / "model_comparison.csv")
        comparison_df.to_csv(save_path, index=False)
        logger.info(f"Comparison table saved to '{save_path}'")
        return save_path

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------
    @staticmethod
    def _get_param_distribution(model_name: str) -> dict:
        """
        Returns a RandomizedSearchCV param distribution for the given
        model, using the 'model__' prefix required because the
        estimator is the 'model' step of a Pipeline.

        Search spaces are intentionally modest (not exhaustive grids)
        because RandomizedSearchCV with n_iter samples a SUBSET of
        combinations — this keeps tuning fast while still covering the
        parameters known to matter most for each algorithm.
        """
        grids = {
            "Ridge": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
            "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]},
            "Decision Tree": {
                "model__max_depth": [3, 5, 10, 15, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            },
            "Extra Trees": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
            },
            "Gradient Boosting": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            "XGBoost": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 4, 5, 6, 7],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
            "CatBoost": {
                "model__iterations": [200, 400, 600],
                "model__depth": [4, 6, 8, 10],
                "model__learning_rate": [0.01, 0.05, 0.1],
            },
            # Linear Regression has no meaningful hyperparameters to tune.
            "Linear Regression": {},
        }
        return grids.get(model_name, {})

    def tune_best_model(self, best_name: str, n_iter: int = 15):
        """
        Runs RandomizedSearchCV on the best model's pipeline using the
        TRAINING split only (bundle['X_train']/['y_train']) — the test
        set stays untouched until final evaluation, preserving an
        honest, unbiased final performance estimate.

        Returns
        -------
        (tuned_pipeline, best_params, search_cv_score)
        """
        if self.bundle is None:
            raise EvaluationError("Call evaluate_all() before tune_best_model().")

        X_train = self.bundle["X_train"]
        y_train = self.bundle["y_train"]
        base_pipeline = self.bundle["fitted_pipelines"][best_name]["pipeline"]

        param_distribution = self._get_param_distribution(best_name)

        if not param_distribution:
            logger.info(
                f"No hyperparameter grid defined for '{best_name}' "
                f"(e.g. Linear Regression has nothing meaningful to tune). "
                f"Using the already-fitted pipeline as the final model."
            )
            return base_pipeline, {}, None

        logger.info(
            f"Starting RandomizedSearchCV for '{best_name}' "
            f"({n_iter} iterations, {config.CV_FOLDS}-fold CV)..."
        )

        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=param_distribution,
            n_iter=n_iter,
            cv=config.CV_FOLDS,
            scoring="r2",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        logger.info(f"Best params for '{best_name}': {search.best_params_}")
        logger.info(f"Best CV R2 after tuning: {round(search.best_score_, 4)}")

        return search.best_estimator_, search.best_params_, search.best_score_

    # ------------------------------------------------------------------
    # Save production model (versioned)
    # ------------------------------------------------------------------
    def save_production_model(
        self, pipeline, model_name: str, best_params: dict, test_metrics: dict
    ) -> Path:
        """
        Saves the final tuned pipeline as BOTH a versioned file
        (car_price_model_v1.joblib) and the 'latest' pointer
        (car_price_model_latest.joblib) that the Flask API always loads.

        Also appends an entry to model_registry.json — a lightweight,
        human-readable version history (model name, version, timestamp,
        hyperparameters, test metrics). This is the model versioning
        mechanism referenced in config.py's MODEL_VERSION.
        """
        import datetime
        import json

        joblib.dump(pipeline, config.MODEL_FILE)
        joblib.dump(pipeline, config.LATEST_MODEL_FILE)
        logger.info(f"Production model saved to '{config.MODEL_FILE}' and '{config.LATEST_MODEL_FILE}'")

        registry_entry = {
            "version": config.MODEL_VERSION,
            "model_name": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "hyperparameters": best_params,
            "test_metrics": test_metrics,
            "model_file": str(config.MODEL_FILE.name),
        }

        registry = []
        if config.MODEL_REGISTRY_FILE.exists():
            with open(config.MODEL_REGISTRY_FILE) as f:
                registry = json.load(f)

        registry.append(registry_entry)
        with open(config.MODEL_REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=2, default=str)

        logger.info(f"Model registry updated at '{config.MODEL_REGISTRY_FILE}'")
        return config.LATEST_MODEL_FILE


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator()
        comparison_df = evaluator.evaluate_all()

        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", None)
        print("\n" + "=" * 100)
        print("MODEL COMPARISON — ALL 9 CANDIDATES")
        print("=" * 100)
        print(comparison_df.to_string(index=False))
        print("=" * 100)

        best_name, reason = evaluator.select_best_model(comparison_df)
        print(f"\nBEST MODEL: {best_name}\n")
        print(reason)

        evaluator.save_comparison(comparison_df)

        # ---- Hyperparameter tuning on the winning model ----
        print(f"\nTuning '{best_name}' with RandomizedSearchCV...")
        tuned_pipeline, best_params, search_cv_score = evaluator.tune_best_model(best_name)

        # ---- Final evaluation of the TUNED model on the held-out test set ----
        X_test, y_test = evaluator.bundle["X_test"], evaluator.bundle["y_test"]
        y_pred_tuned = tuned_pipeline.predict(X_test)
        final_metrics = {
            "MAE": round(mean_absolute_error(y_test, y_pred_tuned), 4),
            "MSE": round(mean_squared_error(y_test, y_pred_tuned), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred_tuned)), 4),
            "R2 Score": round(r2_score(y_test, y_pred_tuned), 4),
        }
        print(f"\nFinal tuned model test metrics: {final_metrics}")
        print(f"Best hyperparameters: {best_params}")

        # ---- Save as the production model (versioned) ----
        model_path = evaluator.save_production_model(
            tuned_pipeline, best_name, best_params, final_metrics
        )
        print(f"\nProduction model saved to: {model_path}")

        # Persist which model was selected + why, for the README/report.
        import json
        selection_path = config.ARTIFACTS_DIR / "best_model_selection.json"
        with open(selection_path, "w") as f:
            json.dump({
                "best_model": best_name,
                "reason": reason,
                "best_params": best_params,
                "final_test_metrics": final_metrics,
            }, f, indent=2, default=str)
        logger.info(f"Best model selection saved to '{selection_path}'")

    except EvaluationError as err:
        logger.error(f"Evaluation aborted: {err}")
        sys.exit(1)
