"""The full season-by-season test: model, market and paper betting.

Everything reported here is "out of sample". A season's forecasts come from a
model that never saw that season, and a season's bets come from an agent that
never saw that season's results.

Paper betting compares four ways of choosing bets on the same games:

* the GRPO agent
* a fixed rule (bet when the edge is at least 3%, quarter-Kelly stake)
* three do-nothing-clever baselines (always the home team, always the
  favourite, always the underdog) that show what the bookmaker's cut costs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import build, evaluate, features
from .agent import GRPOAgent, GRPOConfig, reward_table, state_matrix
from .betting import ThresholdRule, simulate
from .config import ensure, processed_dir

log = logging.getLogger(__name__)

MIN_AGENT_TRAIN_SEASONS = 2
FLAT_STAKE = 0.01


def priced_predictions(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward forecasts for the games that have a market price."""
    merged = preds.merge(odds[["game_id", "home_dec", "away_dec", "market_home_prob", "n_books"]],
                         on="game_id", how="inner")
    return merged.sort_values("date").reset_index(drop=True)


def _states_rewards(rows: pd.DataFrame):
    states = state_matrix(rows["p_model"], rows["home_dec"], rows["away_dec"])
    rewards = reward_table(rows["home_dec"], rows["away_dec"], rows["home_win"])
    return states, rewards


def agent_walk_forward(priced: pd.DataFrame, config: GRPOConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Train the agent on earlier seasons and let it decide on the next one.

    For test season Y the agent trains on priced seasons up to Y-2 and uses
    season Y-1 to choose when to stop training.
    """
    seasons = sorted(int(s) for s in priced["season"].unique())
    decisions, notes = [], {}
    for test_season in seasons:
        earlier = [s for s in seasons if s < test_season]
        if len(earlier) < MIN_AGENT_TRAIN_SEASONS + 1:
            continue
        val_season = earlier[-1]
        train = priced[priced["season"].isin(earlier[:-1])]
        val = priced[priced["season"] == val_season]
        test = priced[priced["season"] == test_season]
        agent = GRPOAgent(config or GRPOConfig())
        agent.fit(*_states_rewards(train), *_states_rewards(val))
        choice = agent.decide(test["p_model"], test["home_dec"], test["away_dec"])
        choice.index = test.index
        decisions.append(choice)
        notes[test_season] = {
            "train_seasons": [int(s) for s in earlier[:-1]], "validation_season": int(val_season),
            "rounds_run": agent.info["rounds_run"],
            "best_validation_reward": agent.info["best_validation_reward"],
            "bet_rate": float((choice["side"] != "pass").mean()),
        }
        log.info("agent %d: trained on %s, bet on %.1f%% of games", test_season, earlier[:-1],
                 100 * notes[test_season]["bet_rate"])
    if not decisions:
        return pd.DataFrame(columns=["side", "stake", "price"]), notes
    return pd.concat(decisions).sort_index(), notes


def _bets(rows: pd.DataFrame, side, stake) -> pd.DataFrame:
    return pd.DataFrame({
        "date": rows["date"].to_numpy(), "season": rows["season"].to_numpy(),
        "side": np.asarray(side), "stake": np.asarray(stake, dtype=float),
        "home_dec": rows["home_dec"].to_numpy(), "away_dec": rows["away_dec"].to_numpy(),
        "home_win": rows["home_win"].to_numpy(),
    })


def strategy_bets(rows: pd.DataFrame, agent_choice: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """The decisions each strategy would have made on ``rows``."""
    out = {}
    if agent_choice is not None and len(agent_choice):
        out["grpo_agent"] = _bets(rows, agent_choice["side"], agent_choice["stake"])
    rule = ThresholdRule().decide(rows["p_model"], rows["home_dec"], rows["away_dec"])
    out["edge_rule"] = _bets(rows, rule["side"], rule["stake"])
    flat = np.full(len(rows), FLAT_STAKE)
    favourite_home = rows["market_home_prob"].to_numpy() >= 0.5
    out["always_home"] = _bets(rows, np.full(len(rows), "home"), flat)
    out["always_favourite"] = _bets(rows, np.where(favourite_home, "home", "away"), flat)
    out["always_underdog"] = _bets(rows, np.where(favourite_home, "away", "home"), flat)
    return out


def _trim(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "bankroll_by_day"}


def betting_report(priced: pd.DataFrame, config: GRPOConfig | None = None) -> dict:
    """Paper results for every strategy, overall and by season."""
    choice, notes = agent_walk_forward(priced, config)
    rows = priced.loc[choice.index] if len(choice) else priced.iloc[0:0]
    if rows.empty:
        return {"seasons": [], "note": "not enough seasons with odds to test the betting strategies"}
    bets = strategy_bets(rows, choice)
    report = {"seasons": sorted(int(s) for s in rows["season"].unique()), "games": int(len(rows)),
              "agent_training": notes, "overall": {}, "by_season": {}, "bankroll_paths": {}}
    for name, frame in bets.items():
        run = simulate(frame)
        report["overall"][name] = _trim(run)
        report["bankroll_paths"][name] = run["bankroll_by_day"]
        for season, part in frame.groupby("season"):
            report["by_season"].setdefault(int(season), {})[name] = _trim(simulate(part))
    return report


def run(*, seed: int = 7, save_to: Path | None = None) -> dict:
    """Run the whole test on the built data and return every result."""
    tables = build.load()
    frame = features.build_features(tables["games"], tables["team_games"], tables["pitcher_games"])
    preds = evaluate.walk_forward(frame, seed=seed, log=log)
    forecast = evaluate.summarize(preds, tables["odds"])
    priced = priced_predictions(preds, tables["odds"])
    betting = betting_report(priced) if len(priced) else {"seasons": []}
    preds.merge(tables["odds"], on="game_id", how="left").to_parquet(
        ensure(processed_dir()) / "walk_forward.parquet", index=False)
    result = {
        "test_seasons": sorted(int(s) for s in preds["season"].unique()),
        "games_tested": int(len(preds)),
        "inputs": list(features.FEATURES),
        "forecast": forecast,
        "betting": betting,
    }
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(result, indent=1, default=float))
    return result
