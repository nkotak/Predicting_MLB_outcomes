"""Paths and constants shared by every stage."""

from __future__ import annotations

import os
from pathlib import Path

# First season of game history. It is used to warm up the running statistics,
# so the first season that models are trained on is one year later.
FIRST_SEASON = 2010

# Retrosheet publishes a season's files in the winter after it ends. Seasons up
# to and including this one are read from Retrosheet. Later seasons, and any
# game that has not been played yet, come from MLB's public schedule feed.
LAST_RETROSHEET_SEASON = 2025

# First season for which ESPN's public feed carries real moneylines.
FIRST_ODDS_SEASON = 2014

USER_AGENT = "mlb-predict/1.0 (python-requests)"


def data_dir() -> Path:
    """Root folder for downloaded and derived data.

    Defaults to ``./data`` under the current directory. Set ``MLB_DATA_DIR`` to
    keep the data somewhere else.
    """
    return Path(os.environ.get("MLB_DATA_DIR", "data")).resolve()


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def models_dir() -> Path:
    return Path(os.environ.get("MLB_MODELS_DIR", "models")).resolve()


def predictions_path() -> Path:
    return data_dir() / "predictions" / "prediction_log.csv"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
