"""The win probability model.

Two simple models are averaged and then calibrated:

* logistic regression, which draws one straight line through the inputs
* gradient boosted trees, which can pick up bends and combinations

Calibration is the last step. It is fitted on a season the two models did not
train on, and it stretches or squeezes their output so that a stated 60%
chance comes true about 60% of the time. That property matters more than raw
accuracy when the output is compared with betting prices.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .features import FEATURES, model_matrix

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _linear() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=2000),
    )


def _trees(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=200,
        l2_regularization=5.0, early_stopping=False, random_state=seed,
    )


@dataclass
class WinModel:
    """Home win probability from the allowed feature columns."""

    seed: int = 7
    features: tuple[str, ...] = FEATURES
    info: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, calibrate: pd.DataFrame) -> "WinModel":
        """Fit on ``train`` and calibrate on ``calibrate``.

        Both frames come from ``features.build_features``. Rows without a
        result (ties, unplayed games) are ignored.
        """
        train = train[train["home_win"].notna()]
        calibrate = calibrate[calibrate["home_win"].notna()]
        if len(train) < 2000 or len(calibrate) < 500:
            raise ValueError(
                f"not enough games to fit: {len(train)} to train on and {len(calibrate)} to calibrate on"
            )
        x, y = model_matrix(train), train["home_win"].astype(int).to_numpy()
        self.linear_ = _linear().fit(x, y)
        self.trees_ = _trees(self.seed).fit(x, y)
        z = self._blend_logit(calibrate)
        self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(
            z.reshape(-1, 1), calibrate["home_win"].astype(int).to_numpy()
        )
        self.info = {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "features": list(self.features),
            "train_seasons": sorted(int(s) for s in train["season"].unique()),
            "calibration_seasons": sorted(int(s) for s in calibrate["season"].unique()),
            "train_games": int(len(train)),
            "calibration_games": int(len(calibrate)),
        }
        return self

    def components(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Uncalibrated probability from each part, for reporting."""
        x = model_matrix(frame)
        return {
            "linear": self.linear_.predict_proba(x)[:, 1],
            "trees": self.trees_.predict_proba(x)[:, 1],
        }

    def _blend_logit(self, frame: pd.DataFrame) -> np.ndarray:
        parts = self.components(frame)
        return 0.5 * (logit(parts["linear"]) + logit(parts["trees"]))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated chance that the home team wins, one value per row."""
        z = self._blend_logit(frame)
        return self.calibrator_.predict_proba(z.reshape(-1, 1))[:, 1]

    # -- saving and loading -------------------------------------------------

    def save(self, folder: Path, name: str = "win_model") -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.joblib"
        joblib.dump(self, path)
        (folder / f"{name}.json").write_text(json.dumps(self.info, indent=2))
        return path

    @staticmethod
    def load(folder: Path, name: str = "win_model") -> "WinModel":
        path = folder / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"no saved model at {path}. Run `mlb train` first.")
        model = joblib.load(path)
        if tuple(model.features) != tuple(FEATURES):
            raise RuntimeError(
                "the saved model was trained on a different set of inputs than this version "
                "of the code produces. Run `mlb train` again."
            )
        return model
