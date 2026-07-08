"""
src/data_ingestion.py
======================
Responsible for ONE thing only: loading the raw dataset safely and
handing back a validated pandas DataFrame.

WHY A SEPARATE INGESTION MODULE?
---------------------------------
In production ML systems, "load the CSV" is never just `pd.read_csv()`.
It needs to:
  1. Confirm the file actually exists (fail fast with a clear message).
  2. Validate the schema BEFORE any downstream code touches it, so a
     malformed dataset doesn't cause a cryptic KeyError three files later.
  3. Log what happened (row count, missing columns, etc.) so failures
     are debuggable from logs alone, without re-running code.
  4. Be swappable — today it reads a local CSV, tomorrow it could read
     from S3, a database, or an API. Because ingestion is isolated in
     its own class, only THIS file would need to change.

DESIGN PATTERN
---------------
We use a class (`DataIngestion`) rather than a bare function so that:
  - State (like the loaded DataFrame or file path) is encapsulated.
  - It's trivially testable (see tests/test_data_ingestion.py later).
  - It mirrors how real ML platforms (e.g. sklearn Pipelines, Kubeflow
    components) structure ingestion as a discrete, reusable unit.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ----------------------------------------------------------------------
# LOGGER SETUP
# ----------------------------------------------------------------------
# Every module gets its own named logger (__name__), but they all share
# the same format/handlers defined once in config.py's LOG_FORMAT.
# This means log lines are traceable to the exact file that emitted them,
# e.g.: "2026-07-04 ... | src.data_ingestion | INFO | Loaded 301 rows"
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(config.LOG_LEVEL)

    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class DataIngestionError(Exception):
    """
    Custom exception for ingestion failures.

    WHY A CUSTOM EXCEPTION?
    A bare `Exception` or `FileNotFoundError` doesn't tell the caller
    WHERE in the pipeline something broke. By raising `DataIngestionError`
    specifically, calling code (train.py, the Flask app, tests) can catch
    ingestion problems distinctly from, say, a model-training failure.
    """
    pass


class DataIngestion:
    """
    Loads and validates the raw car dataset.

    Usage
    -----
    >>> ingestion = DataIngestion()
    >>> df = ingestion.load_data()
    """

    def __init__(self, file_path: Optional[Path] = None):
        """
        Parameters
        ----------
        file_path : Optional[Path]
            Path to the raw CSV. Defaults to config.RAW_DATA_FILE so
            callers don't need to know the path at all — but tests can
            override it to point at a small fixture CSV.
        """
        self.file_path: Path = Path(file_path) if file_path else config.RAW_DATA_FILE
        self._df: Optional[pd.DataFrame] = None

    def _validate_file_exists(self) -> None:
        """Fail fast with a human-readable error if the CSV is missing."""
        if not self.file_path.exists():
            message = (
                f"Dataset not found at '{self.file_path}'. "
                f"Download the CarDekho dataset from Kaggle and place it "
                f"at this path before running ingestion."
            )
            logger.error(message)
            raise DataIngestionError(message)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Confirm the dataframe has (at least) the expected target column
        and is not empty. We deliberately do NOT hard-require every
        single column in config.RAW_COLUMNS to match exactly, because
        Kaggle dataset variants differ slightly in column naming
        (e.g. 'Kms_Driven' vs 'km_driven'). Strict feature-level
        validation happens later in feature_engineering.py once columns
        are normalized.
        """
        if df.empty:
            message = "Loaded dataset is empty (0 rows)."
            logger.error(message)
            raise DataIngestionError(message)

        # Normalize column names for a case/whitespace-insensitive check
        normalized_cols = {c.strip().lower() for c in df.columns}
        target_normalized = config.TARGET_COLUMN.strip().lower()

        if target_normalized not in normalized_cols:
            message = (
                f"Target column '{config.TARGET_COLUMN}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
            logger.error(message)
            raise DataIngestionError(message)

    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV into a DataFrame, validate it, and cache it.

        Returns
        -------
        pd.DataFrame
            The raw, validated dataset.

        Raises
        ------
        DataIngestionError
            If the file is missing, empty, or fails schema validation.
        """
        try:
            self._validate_file_exists()

            logger.info(f"Loading dataset from '{self.file_path}'")
            df = pd.read_csv(self.file_path)

            self._validate_schema(df)

            logger.info(
                f"Dataset loaded successfully: {df.shape[0]} rows, "
                f"{df.shape[1]} columns."
            )
            logger.info(f"Columns found: {list(df.columns)}")

            self._df = df
            return df

        except DataIngestionError:
            # Already logged with a clear message — re-raise as-is so
            # callers can catch this specific exception type.
            raise
        except pd.errors.ParserError as e:
            message = f"Failed to parse CSV (malformed file): {e}"
            logger.error(message)
            raise DataIngestionError(message) from e
        except Exception as e:
            # Catch-all so ingestion NEVER crashes the whole app with an
            # unhandled traceback — it always surfaces as DataIngestionError.
            message = f"Unexpected error while loading dataset: {e}"
            logger.exception(message)
            raise DataIngestionError(message) from e

    def get_basic_info(self) -> dict:
        """
        Return a small summary dict — useful for logging, the EDA
        notebook, and a future '/health' style data-quality check.
        """
        if self._df is None:
            raise DataIngestionError("No data loaded yet. Call load_data() first.")

        return {
            "num_rows": self._df.shape[0],
            "num_columns": self._df.shape[1],
            "columns": list(self._df.columns),
            "missing_values": self._df.isnull().sum().to_dict(),
            "duplicate_rows": int(self._df.duplicated().sum()),
            "dtypes": self._df.dtypes.astype(str).to_dict(),
        }


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Convenience function wrapping DataIngestion for simple one-line use
    in scripts/notebooks that don't need the class's extra methods.

    Example
    -------
    >>> from src.data_ingestion import load_raw_data
    >>> df = load_raw_data()
    """
    return DataIngestion(file_path=file_path).load_data()


if __name__ == "__main__":
    # Allows running this file directly for a quick sanity check:
    #   python src/data_ingestion.py
    try:
        ingestion = DataIngestion()
        data = ingestion.load_data()
        print(data.head())
        print("\nBasic Info:")
        for key, value in ingestion.get_basic_info().items():
            print(f"  {key}: {value}")
    except DataIngestionError as err:
        logger.error(f"Ingestion failed: {err}")
        sys.exit(1)
