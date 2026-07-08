"""
src/visualize_results.py
===========================
Generates two diagnostic plots for the final production model:

  1. FEATURE IMPORTANCE — which inputs actually drive the model's
     predictions. Critical for explaining model behavior to
     non-technical stakeholders and for catching surprises (e.g. if
     'seats' turned out to matter more than 'engine', that's worth
     investigating before trusting the model).

  2. RESIDUAL ANALYSIS — plots (predicted - actual) errors against
     predicted values. A well-behaved model shows residuals randomly
     scattered around zero with no pattern. A funnel shape (variance
     increasing with price) suggests the model is less reliable for
     expensive cars; a curved pattern suggests it's missing a
     non-linear relationship entirely.

WHY THIS IS A SEPARATE FILE FROM evaluate.py
-------------------------------------------------
evaluate.py answers "which model is best and by how much" (metrics).
This file answers "why does the winning model behave the way it does"
(interpretability). Keeping them separate means you can regenerate
diagnostic plots without re-running the (more expensive) full
evaluation + tuning cycle.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers/CI without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


class VisualizationError(Exception):
    """Raised when a saved model/bundle can't be loaded or plotted."""
    pass


class ResultsVisualizer:
    """
    Loads the production model (models/car_price_model_latest.joblib)
    and the held-out test set (from artifacts/training_bundle.joblib)
    to produce feature importance and residual plots.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        bundle_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else config.LATEST_MODEL_FILE
        self.bundle_path = Path(bundle_path) if bundle_path else (
            config.ARTIFACTS_DIR / "training_bundle.joblib"
        )
        self.pipeline = None
        self.X_test = None
        self.y_test = None

    def load(self) -> None:
        if not self.model_path.exists():
            message = f"No production model found at '{self.model_path}'. Run src.evaluate first."
            logger.error(message)
            raise VisualizationError(message)
        if not self.bundle_path.exists():
            message = f"No training bundle found at '{self.bundle_path}'. Run src.train first."
            logger.error(message)
            raise VisualizationError(message)

        self.pipeline = joblib.load(self.model_path)
        bundle = joblib.load(self.bundle_path)
        self.X_test = bundle["X_test"]
        self.y_test = bundle["y_test"]
        logger.info("Model and test set loaded for visualization.")

    # ------------------------------------------------------------------
    def plot_feature_importance(self, top_n: int = 15, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plots the top N most important features, using human-readable
        names from the Preprocessor (not sklearn's cryptic defaults).

        NOTE: not every model type exposes feature importances the
        same way — linear models expose `.coef_` (magnitude of each
        feature's effect), tree/boosting models expose
        `.feature_importances_` (impurity/gain-based importance).
        These aren't directly comparable in scale across model types,
        but within a single model they correctly rank relative
        importance, which is what this plot is for.
        """
        if self.pipeline is None:
            self.load()

        model = self.pipeline.named_steps["model"]
        preprocessor = self.pipeline.named_steps["preprocessing"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_label = "Feature Importance (impurity/gain-based)"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
            importance_label = "Absolute Coefficient Magnitude"
        else:
            logger.warning(
                f"Model type '{type(model).__name__}' exposes neither "
                f"feature_importances_ nor coef_ — skipping importance plot."
            )
            return None

        importance_df = pd.DataFrame({
            "feature": feature_names, "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)

        plt.figure(figsize=(9, max(5, top_n * 0.35)))
        plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#2E86AB")
        plt.xlabel(importance_label)
        plt.title(f"Top {top_n} Feature Importances — {type(model).__name__}")
        plt.tight_layout()

        save_path = Path(save_path) if save_path else (config.ARTIFACTS_DIR / "feature_importance.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Feature importance plot saved to '{save_path}'")
        return save_path

    # ------------------------------------------------------------------
    def plot_residuals(self, save_path: Optional[Path] = None) -> Path:
        """
        Two-panel residual diagnostic:
          Left:  Residuals vs Predicted — checks for patterns/funneling
          Right: Histogram of residuals — checks they're roughly
                 centered at zero and normally distributed (a
                 well-calibrated model's errors should look like noise,
                 not a skewed or bimodal distribution).
        """
        if self.pipeline is None:
            self.load()

        y_pred = self.pipeline.predict(self.X_test)
        residuals = self.y_test.values - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].scatter(y_pred, residuals, alpha=0.5, color="#2E86AB", edgecolors="none")
        axes[0].axhline(0, color="#D64545", linestyle="--", linewidth=1.5)
        axes[0].set_xlabel("Predicted Selling Price")
        axes[0].set_ylabel("Residual (Actual - Predicted)")
        axes[0].set_title("Residuals vs Predicted Values")

        axes[1].hist(residuals, bins=30, color="#2E86AB", edgecolor="white")
        axes[1].axvline(0, color="#D64545", linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Residuals")

        plt.tight_layout()

        save_path = Path(save_path) if save_path else (config.ARTIFACTS_DIR / "residual_analysis.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Residual analysis plot saved to '{save_path}'")
        return save_path


if __name__ == "__main__":
    try:
        visualizer = ResultsVisualizer()
        visualizer.load()

        importance_path = visualizer.plot_feature_importance()
        residual_path = visualizer.plot_residuals()

        print(f"Feature importance plot: {importance_path}")
        print(f"Residual analysis plot: {residual_path}")

    except VisualizationError as err:
        logger.error(f"Visualization failed: {err}")
        sys.exit(1)