"""The win probability model.

Simple models are averaged and then calibrated:

* logistic regression, which draws one straight line through the inputs
* gradient boosted trees, which can pick up bends and combinations
* a run margin model, which is taught the size of each win and not just who
  won. A 9-1 win says more about two teams than a 2-1 win, and a model that
  only sees "won" or "lost" throws that away
* a price-taught model, which learns from what the betting market's closing
  price said about *past* games. One game's result is mostly luck: a team
  with a 60% chance still loses four times in ten. The closing price is a
  far steadier answer key for "how likely was this?". This part is used when
  enough past games have prices. It is never shown the price of the game it
  is predicting: prices are a training target here, not an input

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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .inputs import MODEL_FEATURES, TEACHER

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


MAX_MARGIN = 8.0  # a blowout counts as an eight-run win, so one freak score cannot dominate
MIN_TEACHER_GAMES = 4000  # past games with a closing price needed before the price-taught part is used


def _price_linear() -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30.0))


def _price_trees(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.03, max_iter=400, min_samples_leaf=100,
        l2_regularization=2.0, early_stopping=False, random_state=seed,
    )


def _margin_linear() -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=300.0))


def _margin_trees(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.03, max_iter=250, min_samples_leaf=200,
        l2_regularization=5.0, early_stopping=False, random_state=seed,
    )


@dataclass
class WinModel:
    """Home win probability from the allowed input columns."""

    seed: int = 7
    features: tuple[str, ...] = MODEL_FEATURES
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
        x, y = self._matrix(train), train["home_win"].astype(int).to_numpy()
        self.linear_ = _linear().fit(x, y)
        self.trees_ = _trees(self.seed).fit(x, y)
        margin = (train["home_score"] - train["away_score"]).astype(float).clip(-MAX_MARGIN, MAX_MARGIN).to_numpy()
        self.margin_linear_ = _margin_linear().fit(x, margin)
        self.margin_trees_ = _margin_trees(self.seed).fit(x, margin)
        # Turn an expected run margin into a win chance.
        self.margin_scale_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._margin(x).reshape(-1, 1), y)
        # The price-taught part: fitted to the closing market chance of past games.
        self.price_linear_ = self.price_trees_ = None
        priced = train[train[TEACHER].between(0.02, 0.98)] if TEACHER in train.columns else train.iloc[0:0]
        if len(priced) >= MIN_TEACHER_GAMES:
            xp, target = self._matrix(priced), logit(priced[TEACHER])
            self.price_linear_ = _price_linear().fit(xp, target)
            self.price_trees_ = _price_trees(self.seed).fit(xp, target)
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
            "price_taught_games": int(len(priced)) if self.price_linear_ is not None else 0,
            "calibration_games": int(len(calibrate)),
        }
        return self

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The columns this model may see, and nothing else."""
        return frame.loc[:, list(self.features)].astype(float)

    def _margin(self, x: pd.DataFrame) -> np.ndarray:
        """Expected home run margin: the average of a straight-line and a tree estimate."""
        return 0.5 * (self.margin_linear_.predict(x) + self.margin_trees_.predict(x))

    def components(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Uncalibrated probability from each part, for reporting."""
        x = self._matrix(frame)
        parts = {
            "linear": self.linear_.predict_proba(x)[:, 1],
            "trees": self.trees_.predict_proba(x)[:, 1],
            "margin": self.margin_scale_.predict_proba(self._margin(x).reshape(-1, 1))[:, 1],
        }
        if getattr(self, "price_linear_", None) is not None:
            parts["prices"] = sigmoid(0.5 * (self.price_linear_.predict(x) + self.price_trees_.predict(x)))
        return parts

    def _blend_logit(self, frame: pd.DataFrame) -> np.ndarray:
        """The three parts taught by results count for half, the price-taught part for the other half."""
        parts = self.components(frame)
        results = (logit(parts["linear"]) + logit(parts["trees"]) + logit(parts["margin"])) / 3.0
        if "prices" not in parts:
            return results
        return 0.5 * results + 0.5 * logit(parts["prices"])

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
        if tuple(model.features) != tuple(MODEL_FEATURES):
            raise RuntimeError(
                "the saved model was trained on a different set of inputs than this version "
                "of the code produces. Run `mlb train` again."
            )
        return model
