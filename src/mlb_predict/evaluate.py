"""Honest scoring.

The only test that counts is the one a model would face in real use: it is
fitted on past seasons and then scored on a later season it has never seen.
``walk_forward`` repeats that for every season, one after another.

For a test season Y the model trains on seasons up to Y-2, is calibrated on
season Y-1, and is scored on season Y. The first season in the data only warms
up the running statistics and is never trained or scored on.

Scores are reported next to simple yardsticks: always picking the home team,
the running team rating on its own, and, where odds exist, the betting market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .model import EPS, WinModel, logit, sigmoid

MIN_TRAIN_SEASONS = 3


def log_loss_each(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Average gap between stated chance and what happened, weighted by games.

    Ranges of probability that contain no games are skipped.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def score(y, p) -> dict:
    """Standard scores for a set of probability forecasts."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return {"games": 0}
    accuracy = float(((p > 0.5) == (y == 1)).mean())
    out = {
        "games": int(n),
        "accuracy": accuracy,
        "accuracy_margin": float(1.96 * np.sqrt(max(accuracy * (1 - accuracy), 1e-9) / n)),
        "log_loss": float(log_loss_each(y, p).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "calibration_error": calibration_error(y, p),
        "home_win_rate": float(y.mean()),
    }
    out["auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < n else float("nan")
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, draws: int = 2000, seed: int = 11) -> dict:
    """Is the average of ``a`` really different from the average of ``b``?

    Returns the difference in means (a minus b) and a 95% range from
    resampling the games.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    if n == 0:
        return {"diff": float("nan"), "low": float("nan"), "high": float("nan")}
    idx = rng.integers(0, n, size=(draws, n))
    means = diff[idx].mean(axis=1)
    return {"diff": float(diff.mean()), "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975))}


def _elo_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(
        train[["d_elo"]].to_numpy(), train["home_win"].astype(int).to_numpy())
    return fit.predict_proba(test[["d_elo"]].to_numpy())[:, 1]


def walk_forward(frame: pd.DataFrame, *, seed: int = 7, log=None,
                 features: tuple[str, ...] | None = None,
                 test_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Out-of-sample forecasts for every season that can be tested.

    Returns one row per finished, non-tied game in a test season with the
    forecast of each model part, the final model and the yardsticks.

    ``test_frame`` supplies the rows to forecast when they should differ from
    the rows trained on: the same games described with less information, for
    example with lineups guessed that morning.
    """
    played = frame[frame["home_win"].notna()]
    scored = played if test_frame is None else test_frame[test_frame["home_win"].notna()]
    seasons = sorted(int(s) for s in played["season"].unique())
    first_train = seasons[0] + 1
    out = []
    for test_season in seasons:
        train_seasons = [s for s in seasons if first_train <= s <= test_season - 2]
        if len(train_seasons) < MIN_TRAIN_SEASONS or (test_season - 1) not in seasons:
            continue
        train = played[played["season"].isin(train_seasons)]
        calib = played[played["season"] == test_season - 1]
        test = scored[scored["season"] == test_season]
        model = (WinModel(seed=seed) if features is None else WinModel(seed=seed, features=tuple(features))).fit(train, calib)
        parts = model.components(test)
        fit_rows = pd.concat([train, calib])
        result = test[["game_id", "date", "season", "home", "away", "home_win"]].copy()
        result["p_model"] = model.predict(test)
        result["p_linear"] = parts["linear"]
        result["p_trees"] = parts["trees"]
        result["p_margin"] = parts["margin"]
        result["p_prices"] = parts.get("prices", np.nan)
        result["p_elo"] = _elo_only(fit_rows, test)
        result["p_base_rate"] = float(fit_rows["home_win"].mean())
        out.append(result)
        if log:
            log.info("walk-forward %d: trained on %d-%d, calibrated on %d, %d test games",
                     test_season, train_seasons[0], train_seasons[-1], test_season - 1, len(test))
    if not out:
        raise ValueError("not enough seasons to run a walk-forward test")
    return pd.concat(out, ignore_index=True)


def summarize(preds: pd.DataFrame, odds: pd.DataFrame | None = None) -> dict:
    """Per-season and overall scores for the model and each yardstick."""
    preds = preds.copy()
    if odds is not None and len(odds):
        preds = preds.merge(odds[["game_id", "market_home_prob", "home_dec", "away_dec", "n_books"]],
                            on="game_id", how="left")
    else:
        preds["market_home_prob"] = np.nan
    columns = {"model": "p_model", "linear_only": "p_linear", "trees_only": "p_trees",
               "team_rating_only": "p_elo", "always_home": "p_base_rate"}
    if "p_margin" in preds.columns:
        columns["margin_only"] = "p_margin"
    if "p_team_only" in preds.columns:  # the model without any player-level detail
        columns["team_only"] = "p_team_only"

    def block(rows: pd.DataFrame) -> dict:
        y = rows["home_win"].to_numpy(dtype=float)
        entry = {name: score(y, rows[col].to_numpy()) for name, col in columns.items()}
        priced = rows[rows["market_home_prob"].notna()]
        entry["odds_coverage"] = float(len(priced) / len(rows)) if len(rows) else 0.0
        if len(priced):
            yp = priced["home_win"].to_numpy(dtype=float)
            entry["market"] = score(yp, priced["market_home_prob"].to_numpy())
            entry["model_on_priced_games"] = score(yp, priced["p_model"].to_numpy())
        return entry

    by_season = {int(s): block(rows) for s, rows in preds.groupby("season")}
    overall = block(preds)
    if "p_prices" in preds.columns and preds["p_prices"].notna().any():
        # The price-taught part only exists once enough past games have prices, so it is
        # scored on the seasons that have it, next to the whole model on those same seasons.
        taught = preds[preds["p_prices"].notna()]
        y_taught = taught["home_win"].to_numpy(dtype=float)
        overall["price_taught_only"] = score(y_taught, taught["p_prices"].to_numpy())
        overall["model_on_price_taught_seasons"] = score(y_taught, taught["p_model"].to_numpy())
        overall["price_taught_seasons"] = sorted(int(s) for s in taught["season"].unique())
    priced = preds[preds["market_home_prob"].notna()]
    if len(priced):
        y = priced["home_win"].to_numpy(dtype=float)
        overall["model_minus_market_log_loss"] = paired_bootstrap(
            log_loss_each(y, priced["p_model"].to_numpy()),
            log_loss_each(y, priced["market_home_prob"].to_numpy()))
    y_all = preds["home_win"].to_numpy(dtype=float)
    overall["model_minus_always_home_log_loss"] = paired_bootstrap(
        log_loss_each(y_all, preds["p_model"].to_numpy()),
        log_loss_each(y_all, preds["p_base_rate"].to_numpy()))
    overall["model_minus_always_home_accuracy"] = paired_bootstrap(
        ((preds["p_model"].to_numpy() > 0.5) == (y_all == 1)).astype(float),
        (y_all == 1).astype(float))
    return {"overall": overall, "by_season": by_season}


__all__ = ["walk_forward", "summarize", "score", "paired_bootstrap", "calibration_error",
           "log_loss_each", "logit", "sigmoid"]
