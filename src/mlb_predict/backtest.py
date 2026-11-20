"""The full season-by-season test: model, market and paper betting.

Everything reported here is "out of sample". A season's forecasts come from a
model that never saw that season, and a season's bets come from an agent that
never saw that season's results.

Paper betting compares several ways of choosing bets on the same games:

* the GRPO agent
* a fixed rule (bet when the edge is at least 3%, quarter-Kelly stake)
* three do-nothing-clever baselines (always the home team, always the
  favourite, always the underdog) that show what the bookmaker's cut costs

It is run twice. First at closing prices, the hardest test because the
closing price already contains everything the market learned that day.
Then at opening prices, for the seasons where those exist, to see what an
early bet would have been worth. For early bets the report also shows the
"value against the close": how the price taken compares with the fair closing
price. Beating the close is the standard sign of a real edge, and it shows up
long before profits can be told apart from luck.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import build, evaluate, features
from .agent import GRPOAgent, GRPOConfig, reward_table, state_matrix
from .betting import ThresholdRule, fair_probabilities, simulate
from .config import ensure, processed_dir
from .model import logit

log = logging.getLogger(__name__)

MIN_AGENT_TRAIN_SEASONS = 2
MIN_SEASON_GAMES_WITH_ODDS = 300  # a season with fewer priced games is not used for betting tests
FLAT_STAKE = 0.01
CLOSE = ("home_dec", "away_dec")
OPEN = ("open_home_dec", "open_away_dec")


def priced_predictions(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward forecasts for the games that have a market price."""
    keep = [c for c in ("game_id", "home_dec", "away_dec", "market_home_prob", "n_books",
                        "open_home_dec", "open_away_dec", "open_market_home_prob") if c in odds.columns]
    merged = preds.merge(odds[keep], on="game_id", how="inner")
    return merged.sort_values("date").reset_index(drop=True)


def usable(priced: pd.DataFrame, prices: tuple[str, str]) -> pd.DataFrame:
    """Rows that have this kind of price, in seasons with enough of them."""
    if prices[0] not in priced.columns:
        return priced.iloc[0:0]
    rows = priced[priced[prices[0]].notna() & priced[prices[1]].notna()]
    counts = rows.groupby("season").size()
    return rows[rows["season"].isin(counts[counts >= MIN_SEASON_GAMES_WITH_ODDS].index)]


def _states_rewards(rows: pd.DataFrame, prices: tuple[str, str]):
    home, away = rows[prices[0]], rows[prices[1]]
    return state_matrix(rows["p_model"], home, away), reward_table(home, away, rows["home_win"])


def agent_walk_forward(priced: pd.DataFrame, config: GRPOConfig | None = None, *,
                       prices: tuple[str, str] = CLOSE,
                       min_train_seasons: int = MIN_AGENT_TRAIN_SEASONS) -> tuple[pd.DataFrame, dict]:
    """Train the agent on earlier seasons and let it decide on the next one.

    For test season Y the agent trains on priced seasons up to Y-2 and uses
    season Y-1 to judge whether what it learned holds up.
    """
    rows = usable(priced, prices)
    seasons = sorted(int(s) for s in rows["season"].unique())
    decisions, notes = [], {}
    for test_season in seasons:
        earlier = [s for s in seasons if s < test_season]
        if len(earlier) < min_train_seasons + 1:
            continue
        val_season = earlier[-1]
        train = rows[rows["season"].isin(earlier[:-1])]
        val = rows[rows["season"] == val_season]
        test = rows[rows["season"] == test_season]
        agent = GRPOAgent(config or GRPOConfig())
        agent.fit(*_states_rewards(train, prices), *_states_rewards(val, prices))
        choice = agent.decide(test["p_model"], test[prices[0]], test[prices[1]])
        choice.index = test.index
        decisions.append(choice)
        notes[test_season] = {
            "train_seasons": [int(s) for s in earlier[:-1]], "validation_season": int(val_season),
            "rounds_run": agent.info["rounds_run"],
            "kept_never_bet_policy": agent.info["kept_never_bet_policy"],
            "best_validation_reward": agent.info["best_validation_reward"],
            "bet_rate": float((choice["side"] != "pass").mean()),
        }
        log.info("agent %d: trained on %s, bet on %.1f%% of games", test_season, earlier[:-1],
                 100 * notes[test_season]["bet_rate"])
    if not decisions:
        return pd.DataFrame(columns=["side", "stake", "price"]), notes
    return pd.concat(decisions).sort_index(), notes


def _bets(rows: pd.DataFrame, side, stake, prices: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": rows["date"].to_numpy(), "season": rows["season"].to_numpy(),
        "side": np.asarray(side), "stake": np.asarray(stake, dtype=float),
        "home_dec": rows[prices[0]].to_numpy(), "away_dec": rows[prices[1]].to_numpy(),
        "home_win": rows["home_win"].to_numpy(),
        # The fair closing chance, used to ask "did this bet beat the closing price?"
        "close_home_prob": rows["market_home_prob"].to_numpy(),
    })


def strategy_bets(rows: pd.DataFrame, agent_choice: pd.DataFrame | None = None,
                  prices: tuple[str, str] = CLOSE) -> dict[str, pd.DataFrame]:
    """The decisions each strategy would have made on ``rows``."""
    out = {}
    if agent_choice is not None and len(agent_choice):
        out["grpo_agent"] = _bets(rows, agent_choice["side"], agent_choice["stake"], prices)
    rule = ThresholdRule().decide(rows["p_model"], rows[prices[0]], rows[prices[1]])
    out["edge_rule"] = _bets(rows, rule["side"], rule["stake"], prices)
    flat = np.full(len(rows), FLAT_STAKE)
    fair_home, _ = fair_probabilities(rows[prices[0]], rows[prices[1]])
    favourite_home = np.asarray(fair_home) >= 0.5
    out["always_home"] = _bets(rows, np.full(len(rows), "home"), flat, prices)
    out["always_favourite"] = _bets(rows, np.where(favourite_home, "home", "away"), flat, prices)
    out["always_underdog"] = _bets(rows, np.where(favourite_home, "away", "home"), flat, prices)
    return out


def value_against_close(bets: pd.DataFrame) -> float:
    """Average of (price taken x fair closing chance - 1) over the bets placed.

    Above zero means the bets were made at better prices than the market's
    final, fairest view. That is what "beating the closing line" means.
    """
    placed = bets[bets["side"] != "pass"]
    if placed.empty:
        return float("nan")
    home = (placed["side"] == "home").to_numpy()
    price = np.where(home, placed["home_dec"], placed["away_dec"])
    fair = np.where(home, placed["close_home_prob"], 1.0 - placed["close_home_prob"])
    return float((price * fair - 1.0).mean())


def _summary(frame: pd.DataFrame) -> tuple[dict, list]:
    run = simulate(frame)
    out = {k: v for k, v in run.items() if k != "bankroll_by_day"}
    out["value_against_close"] = value_against_close(frame)
    return out, run["bankroll_by_day"]


def betting_report(priced: pd.DataFrame, config: GRPOConfig | None = None, *,
                   prices: tuple[str, str] = CLOSE,
                   min_train_seasons: int = MIN_AGENT_TRAIN_SEASONS) -> dict:
    """Paper results for every strategy, overall and by season."""
    choice, notes = agent_walk_forward(priced, config, prices=prices, min_train_seasons=min_train_seasons)
    rows = priced.loc[choice.index] if len(choice) else priced.iloc[0:0]
    if rows.empty:
        return {"seasons": [], "note": "not enough seasons with these prices to test the betting strategies"}
    bets = strategy_bets(rows, choice, prices)
    report = {"seasons": sorted(int(s) for s in rows["season"].unique()), "games": int(len(rows)),
              "agent_training": notes, "overall": {}, "by_season": {}, "bankroll_paths": {}}
    for name, frame in bets.items():
        report["overall"][name], report["bankroll_paths"][name] = _summary(frame)
        for season, part in frame.groupby("season"):
            report["by_season"].setdefault(int(season), {})[name] = _summary(part)[0]
    return report


def line_movement(priced: pd.DataFrame) -> dict:
    """Does the model's disagreement with the opening price predict where the price goes?

    For each game: how far the model sits from the opening price, and how far
    the price moved by the close, both on the log-odds scale. A slope above
    zero means the market tends to move toward the model during the day.
    """
    rows = usable(priced, OPEN)
    if rows.empty:
        return {"games": 0}
    opening, closing = logit(rows["open_market_home_prob"]), logit(rows["market_home_prob"])
    gap, move = logit(rows["p_model"]) - opening, closing - opening
    slope = float(np.cov(gap, move)[0, 1] / np.var(gap, ddof=1))
    corr = float(np.corrcoef(gap, move)[0, 1])
    y = rows["home_win"].to_numpy(dtype=float)
    return {
        "games": int(len(rows)), "seasons": sorted(int(s) for s in rows["season"].unique()),
        "slope": slope, "correlation": corr,
        "correlation_margin": float(1.96 / np.sqrt(len(rows))),
        "average_move_points": float(100 * (rows["market_home_prob"] - rows["open_market_home_prob"]).abs().mean()),
        "log_loss_opening_price": float(evaluate.log_loss_each(y, rows["open_market_home_prob"].to_numpy()).mean()),
        "log_loss_closing_price": float(evaluate.log_loss_each(y, rows["market_home_prob"].to_numpy()).mean()),
        "log_loss_model": float(evaluate.log_loss_each(y, rows["p_model"].to_numpy()).mean()),
    }


def run(*, seed: int = 7, save_to: Path | None = None) -> dict:
    """Run the whole test on the built data and return every result."""
    tables = build.load()
    frame = features.build_features(tables["games"], tables["team_games"], tables["pitcher_games"])
    preds = evaluate.walk_forward(frame, seed=seed, log=log)
    forecast = evaluate.summarize(preds, tables["odds"])
    priced = priced_predictions(preds, tables["odds"])
    betting = betting_report(priced) if len(priced) else {"seasons": []}
    early = betting_report(priced, prices=OPEN, min_train_seasons=1) if len(priced) else {"seasons": []}
    early["line_movement"] = line_movement(priced) if len(priced) else {"games": 0}
    preds.merge(tables["odds"], on="game_id", how="left").to_parquet(
        ensure(processed_dir()) / "walk_forward.parquet", index=False)
    result = {
        "test_seasons": sorted(int(s) for s in preds["season"].unique()),
        "games_tested": int(len(preds)),
        "inputs": list(features.FEATURES),
        "forecast": forecast,
        "betting": betting,
        "early_prices": early,
    }
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(result, indent=1, default=float))
    return result
