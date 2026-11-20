"""Predictions for a date, the prediction log, grading and continued learning.

Every prediction that is shown can be written to a log with the time it was
made and the price at that moment. After the games finish, ``grade`` fills in
what happened. A dated log that is written before the games is the only real
proof that a model is any good, so it is the basis of the scorecard.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, build, evaluate, features, teams
from .agent import GRPOAgent, GRPOConfig, reward_table, state_matrix
from .betting import ThresholdRule, decimal_to_american, edge, settle, simulate
from .config import models_dir, predictions_path
from .model import WinModel

log = logging.getLogger(__name__)

LOG_COLUMNS = [
    "logged_utc", "game_id", "date", "away", "home", "away_sp", "home_sp", "made_before_game",
    "p_home", "home_dec", "away_dec", "market_home_prob", "n_books",
    "agent_side", "agent_stake", "rule_side", "rule_stake", "model_calibrated_through",
    "graded", "home_win", "agent_profit", "rule_profit", "close_home_dec", "close_away_dec",
]
MIN_SEASON_GAMES = 2000  # a season counts as complete once it has this many finished games


# ---------------------------------------------------------------------------
# Training the model and agent that are used for new predictions
# ---------------------------------------------------------------------------


def _complete_seasons(frame: pd.DataFrame) -> list[int]:
    """Seasons that are over. The current year counts once nearly all its games are in."""
    counts = frame[frame["home_win"].notna()].groupby("season").size()
    this_year = dt.date.today().year
    return sorted(int(s) for s, n in counts.items() if s < this_year or n >= MIN_SEASON_GAMES)


def train(*, seed: int = 7) -> dict:
    """Fit and save the model and the agent on everything available.

    The model trains on every complete season except the latest one, which is
    held back to calibrate it. The agent trains on out-of-sample forecasts for
    every priced season except the latest one, which decides when to stop.
    """
    tables = build.load()
    frame = features.build_features(tables["games"], tables["team_games"], tables["pitcher_games"])
    seasons = _complete_seasons(frame)
    if len(seasons) < 6:
        raise ValueError(f"need at least 6 complete seasons to train, found {len(seasons)}")
    played = frame[frame["home_win"].notna()]
    train_rows = played[played["season"].isin(seasons[1:-1])]  # the first season only warms up
    calib_rows = played[played["season"] == seasons[-1]]
    model = WinModel(seed=seed).fit(train_rows, calib_rows)
    model.save(models_dir())

    preds = evaluate.walk_forward(frame, seed=seed)
    priced = backtest.priced_predictions(preds, tables["odds"])
    priced_seasons = sorted(int(s) for s in priced["season"].unique())
    info = {"model": model.info}
    if len(priced_seasons) >= 3:
        fit_rows = priced[priced["season"].isin(priced_seasons[:-1])]
        val_rows = priced[priced["season"] == priced_seasons[-1]]
        agent = GRPOAgent(GRPOConfig()).fit(
            state_matrix(fit_rows["p_model"], fit_rows["home_dec"], fit_rows["away_dec"]),
            reward_table(fit_rows["home_dec"], fit_rows["away_dec"], fit_rows["home_win"]),
            state_matrix(val_rows["p_model"], val_rows["home_dec"], val_rows["away_dec"]),
            reward_table(val_rows["home_dec"], val_rows["away_dec"], val_rows["home_win"]),
        )
        agent.info.update({"train_seasons": priced_seasons[:-1], "validation_season": priced_seasons[-1]})
        agent.save(models_dir())
        info["agent"] = agent.info
    else:
        log.warning("fewer than 3 seasons have odds, so no agent was trained")
    (models_dir() / "training_info.json").write_text(json.dumps(info, indent=2, default=float))
    return info


def learn(*, rounds: int = 20, window_seasons: int = 4) -> dict:
    """Let the agent keep learning from the newest graded results.

    Starting from its current policy, the agent runs more GRPO rounds on the
    most recent seasons of priced games. The newest quarter of those games is
    held back to decide which round to keep.
    """
    tables = build.load()
    frame = features.build_features(tables["games"], tables["team_games"], tables["pitcher_games"])
    preds = evaluate.walk_forward(frame)
    priced = backtest.priced_predictions(preds, tables["odds"])
    seasons = sorted(int(s) for s in priced["season"].unique())[-window_seasons:]
    recent = priced[priced["season"].isin(seasons)].sort_values("date")
    cut = int(len(recent) * 0.75)
    fit_rows, val_rows = recent.iloc[:cut], recent.iloc[cut:]
    agent = GRPOAgent.load(models_dir())
    val_states = state_matrix(val_rows["p_model"], val_rows["home_dec"], val_rows["away_dec"])
    val_rewards = reward_table(val_rows["home_dec"], val_rows["away_dec"], val_rows["home_win"])
    before = agent.value(val_states, val_rewards)
    agent.fit(
        state_matrix(fit_rows["p_model"], fit_rows["home_dec"], fit_rows["away_dec"]),
        reward_table(fit_rows["home_dec"], fit_rows["away_dec"], fit_rows["home_win"]),
        val_states, val_rewards, warm_start=True, iterations=rounds,
    )
    after = agent.value(val_states, val_rewards)
    agent.save(models_dir())
    entry = {
        "updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "games_learned_from": int(len(fit_rows)), "games_held_back": int(len(val_rows)),
        "newest_game": recent["date"].max().strftime("%Y-%m-%d"),
        "held_back_reward_before": before["mean_reward"], "held_back_reward_after": after["mean_reward"],
        "bet_rate_before": before["bet_rate"], "bet_rate_after": after["bet_rate"],
    }
    history_path = models_dir() / "agent_learning_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(entry)
    history_path.write_text(json.dumps(history, indent=2))
    return entry


# ---------------------------------------------------------------------------
# Predictions for one date
# ---------------------------------------------------------------------------


def predict_date(date: dt.date, *, write_log: bool = True) -> pd.DataFrame:
    """Win chances and paper-bet suggestions for every game on ``date``."""
    tables = build.load()
    games = tables["games"]
    day = pd.Timestamp(date)
    if not (games["date"] == day).any():
        raise ValueError(
            f"no games on {date} in the built data. Run `mlb update` to refresh the schedule, "
            "or check the date."
        )
    frame = features.build_features(games, tables["team_games"], tables["pitcher_games"])
    rows = frame[frame["date"] == day].copy()
    model = WinModel.load(models_dir())
    rows["p_home"] = model.predict(rows)

    names = tables["players"].set_index("player_id")["name"]
    out = rows[["game_id", "date", "away", "home", "status", "home_sp_id", "away_sp_id", "p_home"]].copy()
    out["home_sp"] = out["home_sp_id"].map(names).fillna("not announced")
    out["away_sp"] = out["away_sp_id"].map(names).fillna("not announced")
    out["made_before_game"] = out["status"].eq("scheduled")
    out["fair_home_american"] = [round(decimal_to_american(1.0 / p)) for p in out["p_home"]]
    out["fair_away_american"] = [round(decimal_to_american(1.0 / (1.0 - p))) for p in out["p_home"]]
    out = out.merge(tables["odds"][["game_id", "home_dec", "away_dec", "market_home_prob", "n_books"]],
                    on="game_id", how="left")
    out["edge_home"] = edge(out["p_home"], out["home_dec"])
    out["edge_away"] = edge(1.0 - out["p_home"], out["away_dec"])
    out["agent_side"], out["agent_stake"] = "no price", 0.0
    out["rule_side"], out["rule_stake"] = "no price", 0.0
    priced = out["home_dec"].notna().to_numpy()
    if priced.any():
        part = out[priced]
        rule = ThresholdRule().decide(part["p_home"], part["home_dec"], part["away_dec"])
        out.loc[priced, "rule_side"], out.loc[priced, "rule_stake"] = rule["side"].to_numpy(), rule["stake"].to_numpy()
        try:
            choice = GRPOAgent.load(models_dir()).decide(part["p_home"], part["home_dec"], part["away_dec"])
            out.loc[priced, "agent_side"] = choice["side"].to_numpy()
            out.loc[priced, "agent_stake"] = choice["stake"].to_numpy()
        except FileNotFoundError:
            out.loc[priced, "agent_side"] = "no agent"
    calibrated = model.info.get("calibration_seasons") or [None]
    out["model_calibrated_through"] = calibrated[-1]
    out = out.drop(columns=["home_sp_id", "away_sp_id"]).reset_index(drop=True)
    if write_log:
        append_log(out)
    return out


# ---------------------------------------------------------------------------
# The prediction log
# ---------------------------------------------------------------------------


def read_log(path: Path | None = None) -> pd.DataFrame:
    path = path or predictions_path()
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    frame = pd.read_csv(path, parse_dates=["date"])
    for column in LOG_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["graded"] = frame["graded"].fillna(False).astype(bool)
    frame["made_before_game"] = frame["made_before_game"].astype(bool)
    return frame[LOG_COLUMNS]


def append_log(predictions: pd.DataFrame, path: Path | None = None) -> int:
    path = path or predictions_path()
    rows = predictions.copy()
    rows["logged_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows["graded"] = False
    for column in LOG_COLUMNS:
        if column not in rows.columns:
            rows[column] = np.nan
    existing = read_log(path)
    combined = rows[LOG_COLUMNS] if existing.empty else pd.concat([existing, rows[LOG_COLUMNS]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(rows)


def latest_per_game(frame: pd.DataFrame) -> pd.DataFrame:
    """The last prediction logged for each game. Earlier ones are superseded."""
    return frame.sort_values("logged_utc").drop_duplicates("game_id", keep="last")


def grade(path: Path | None = None) -> dict:
    """Fill in results for logged games that have finished."""
    path = path or predictions_path()
    frame = read_log(path)
    if frame.empty:
        return {"graded_now": 0, "waiting": 0}
    tables = build.load(("games", "odds"))
    final = tables["games"][tables["games"]["status"] == "final"].set_index("game_id")
    closing = tables["odds"].set_index("game_id")
    todo = ~frame["graded"] & frame["game_id"].isin(final.index)
    for index in frame.index[todo]:
        game = final.loc[frame.at[index, "game_id"]]
        if game["home_score"] == game["away_score"]:
            continue
        won = float(game["home_score"] > game["away_score"])
        frame.at[index, "home_win"] = won
        for who in ("agent", "rule"):
            side, stake = frame.at[index, f"{who}_side"], frame.at[index, f"{who}_stake"]
            if side in ("home", "away") and stake > 0:
                profit = settle([side], [stake], [frame.at[index, "home_dec"]], [frame.at[index, "away_dec"]], [won])[0]
                frame.at[index, f"{who}_profit"] = profit / stake  # per unit staked
        if frame.at[index, "game_id"] in closing.index:
            price = closing.loc[frame.at[index, "game_id"]]
            frame.at[index, "close_home_dec"], frame.at[index, "close_away_dec"] = price["home_dec"], price["away_dec"]
        frame.at[index, "graded"] = True
    frame.to_csv(path, index=False)
    return {"graded_now": int(todo.sum()), "waiting": int((~frame["graded"]).sum())}


def scorecard(path: Path | None = None) -> dict:
    """Track record of the logged predictions.

    Predictions made before the game count as the real record. Predictions
    logged after a game finished are replays and are reported separately.
    """
    frame = latest_per_game(read_log(path))
    graded = frame[frame["graded"] & frame["home_win"].notna()]
    out = {"logged_games": int(len(frame)), "graded_games": int(len(graded)),
           "waiting_for_results": int((~frame["graded"]).sum())}
    for label, part in (("made_before_game", graded[graded["made_before_game"]]),
                        ("replayed_after_game", graded[~graded["made_before_game"]])):
        if part.empty:
            out[label] = {"games": 0}
            continue
        y = part["home_win"].to_numpy(dtype=float)
        entry = {"model": evaluate.score(y, part["p_home"].to_numpy(dtype=float))}
        priced = part[part["market_home_prob"].notna()]
        if len(priced):
            entry["market"] = evaluate.score(priced["home_win"].to_numpy(dtype=float),
                                             priced["market_home_prob"].to_numpy(dtype=float))
            for who in ("agent", "rule"):
                bets = pd.DataFrame({
                    "date": priced["date"], "side": priced[f"{who}_side"].where(
                        priced[f"{who}_side"].isin(["home", "away"]), "pass"),
                    "stake": priced[f"{who}_stake"].fillna(0.0), "home_dec": priced["home_dec"],
                    "away_dec": priced["away_dec"], "home_win": priced["home_win"],
                })
                run = simulate(bets)
                entry[f"{who}_paper_bets"] = {k: v for k, v in run.items() if k != "bankroll_by_day"}
        out[label] = entry
    return out


def team_label(code: str) -> str:
    return teams.name(code)
