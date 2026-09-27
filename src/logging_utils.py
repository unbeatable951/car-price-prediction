"""
src/logging_utils.py
======================
Centralized logger construction, used by every module in this project
instead of each one repeating its own logging.FileHandler setup.

WHY THIS EXISTS
----------------
Previously, src/data_ingestion.py, feature_engineering.py,
preprocessing.py, train.py, evaluate.py, predict.py, and
visualize_results.py each duplicated the exact same 10-line block:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(config.LOG_LEVEL)
        file_handler = logging.FileHandler(config.LOG_FILE)
        ...
Besides the duplication, plain `logging.FileHandler` never rotates —
config.LOG_FILE grows forever for the lifetime of a long-running
container or dev machine. `get_logger()` fixes both: one place to
change logging behavior, and rotation so a single log file can't grow
unbounded.

Uses RotatingFileHandler (size-based) rather than TimedRotatingFileHandler
(daily, say) because a low-traffic service might not fill a day's worth
of logs for a long time, while a traffic spike could fill a size-based
file quickly — size-based rotation bounds disk usage predictably
regardless of traffic pattern, which matters more here than aligning
rotation to calendar days.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import config

# 5 MB per file, keep 3 old copies (~20 MB ceiling total for this log).
# Generous enough to not lose useful history on a small deployment
# (Render's free/starter tiers), small enough to never become a disk
# problem on its own.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured identically everywhere it's used:
    same level, same format, same rotating file handler, plus a
    console handler for local/Docker/Render log streaming.

    Safe to call multiple times with the same `name` (e.g. re-importing
    a module during interactive use) — handlers are only attached once,
    the same guard the original per-module code used.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(config.LOG_LEVEL)

        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
        )
        file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger