"""The full season-by-season test: model, market and paper betting.

Everything reported here is "out of sample". A season's forecasts come from a
model that never saw that season, and a season's bets come from an agent that
never saw that season's results.

Paper betting compares several ways of choosing bets on the same games:

* the GRPO agent, which passes unless betting is proven better
* the same agent with that safety bar off, to show what it learned to bet on
* a fixed rule (bet when the edge is at least 3%, quarter-Kelly stake)
* three do-nothing-clever baselines (always the home team, always the
  favourite, always the underdog) that show what the bookmaker's cut costs

It is run three times. First at closing prices, the hardest test because
the closing price already contains everything the market learned that day.
Then at the best closing price any sportsbook offered, on the games that at
least three sportsbooks quoted, to show what shopping around is worth. Then
at opening prices, for the seasons where those exist, to see what an early
bet would have been worth. For early bets the report
also shows the "value against the close": how the price taken compares with
the fair closing price. Beating the close is the standard sign of a real
edge, and it shows up long before profits can be told apart from luck.

Early bets are judged with lineups guessed from recent games, because the
real lineup is not announced until a few hours before first pitch.

The report also answers three questions about the model itself: what the
player-level detail added over the team-only model, what each group of
inputs is worth, and whether the model knows anything the market price does
not already contain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from . import build, evaluate, inputs, players
from .agent import GRPOAgent, GRPOConfig, reward_table, state_matrix
from .betting import ThresholdRule, fair_probabilities, simulate
from .config import ensure, processed_dir
from .model import logit

log = logging.getLogger(__name__)

MIN_AGENT_TRAIN_SEASONS = 2
MIN_SEASON_GAMES_WITH_ODDS = 300  # a season with fewer priced games is not used for betting tests
FLAT_STAKE = 0.01
CLOSE = ("home_dec", "away_dec")
BEST = ("best_home_dec", "best_away_dec")
OPEN = ("open_home_dec", "open_away_dec")
MIN_BOOKS_TO_SHOP = 3  # shopping for a price needs several sportsbooks quoting the game


def priced_predictions(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward forecasts for the games that have a market price."""
    keep = [c for c in ("game_id", "home_dec", "away_dec", "market_home_prob", "n_books",
                        "best_home_dec", "best_away_dec",
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
        # The same games with the safety bar off: what the learned policy would have bet.
        raw = agent.decide(test["p_model"], test[prices[0]], test[prices[1]], guarded=False)
        choice["raw_side"], choice["raw_stake"] = raw["side"].to_numpy(), raw["stake"].to_numpy()
        decisions.append(choice)
        notes[test_season] = {
            "train_seasons": [int(s) for s in earlier[:-1]], "validation_season": int(val_season),
            "rounds_run": agent.info["rounds_run"],
            "kept_never_bet_policy": agent.info["kept_never_bet_policy"],
            "best_validation_reward": agent.info["best_validation_reward"],
            "bet_rate": float((choice["side"] != "pass").mean()),
            "bet_rate_safety_off": float((choice["raw_side"] != "pass").mean()),
            **{k: agent.info[k] for k in ("bet_rate_first_round", "bet_rate_last_round",
                                          "train_reward_first_round", "train_reward_last_round",
                                          "validation_reward_last_round") if k in agent.info},
        }
        log.info("agent %d: trained on %s, bet on %.1f%% of games", test_season, earlier[:-1],
                 100 * notes[test_season]["bet_rate"])
    if not decisions:
        return pd.DataFrame(columns=["side", "stake", "price", "raw_side", "raw_stake"]), notes
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
        if "raw_side" in agent_choice.columns:
            out["grpo_agent_safety_off"] = _bets(rows, agent_choice["raw_side"], agent_choice["raw_stake"], prices)
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


def log_loss_gap(a: pd.DataFrame, b: pd.DataFrame, column: str = "p_model") -> dict:
    """Log loss of forecasts ``a`` minus forecasts ``b`` on the games both cover, with a 95% range."""
    both = a[["game_id", "home_win", column]].merge(b[["game_id", column]], on="game_id", suffixes=("_a", "_b"))
    y = both["home_win"].to_numpy(dtype=float)
    out = evaluate.paired_bootstrap(evaluate.log_loss_each(y, both[f"{column}_a"].to_numpy()),
                                    evaluate.log_loss_each(y, both[f"{column}_b"].to_numpy()))
    out["games"] = int(len(both))
    return out


def market_blend(priced: pd.DataFrame, market: str = "market_home_prob", *, min_train_seasons: int = 2) -> dict:
    """Does the model know anything the market price does not already contain?

    For each season, a simple mix of the market's chance and the model's
    chance is fitted on earlier seasons and scored on that season. If the mix
    scores no better than the market alone, the model adds nothing to it. The
    weight the mix gives the model is reported too: near zero means "ignore
    the model", near one means "the model matters as much as the price".
    """
    if market not in priced.columns:
        return {"games": 0}
    rows = priced[priced[market].notna() & priced["home_win"].notna()]
    counts = rows.groupby("season").size()
    rows = rows[rows["season"].isin(counts[counts >= MIN_SEASON_GAMES_WITH_ODDS].index)]
    seasons = sorted(int(s) for s in rows["season"].unique())
    if len(seasons) < min_train_seasons + 1:
        return {"games": 0}

    def design(part: pd.DataFrame, both: bool) -> np.ndarray:
        columns = [logit(part[market])] + ([logit(part["p_model"])] if both else [])
        return np.column_stack(columns)

    tested = []
    for season in seasons[min_train_seasons:]:
        train, test = rows[rows["season"] < season], rows[rows["season"] == season]
        y_train = train["home_win"].astype(int).to_numpy()
        mix = LogisticRegression(C=1e6, max_iter=1000).fit(design(train, True), y_train)
        alone = LogisticRegression(C=1e6, max_iter=1000).fit(design(train, False), y_train)
        tested.append(pd.DataFrame({
            "y": test["home_win"].to_numpy(dtype=float),
            "p_mix": mix.predict_proba(design(test, True))[:, 1],
            "p_market": alone.predict_proba(design(test, False))[:, 1],
            "p_model": test["p_model"].to_numpy(),
        }))
    scored = pd.concat(tested, ignore_index=True)
    y = scored["y"].to_numpy()
    final = LogisticRegression(C=1e6, max_iter=1000).fit(design(rows, True), rows["home_win"].astype(int).to_numpy())
    weight_market, weight_model = (float(v) for v in final.coef_[0])
    return {
        "games": int(len(scored)), "seasons": seasons[min_train_seasons:],
        "log_loss_market": float(evaluate.log_loss_each(y, scored["p_market"].to_numpy()).mean()),
        "log_loss_model": float(evaluate.log_loss_each(y, scored["p_model"].to_numpy()).mean()),
        "log_loss_mix": float(evaluate.log_loss_each(y, scored["p_mix"].to_numpy()).mean()),
        "mix_minus_market": evaluate.paired_bootstrap(
            evaluate.log_loss_each(y, scored["p_mix"].to_numpy()),
            evaluate.log_loss_each(y, scored["p_market"].to_numpy())),
        "weight_on_market": weight_market, "weight_on_model": weight_model,
        "model_share": float(weight_model / (weight_model + weight_market)) if weight_model + weight_market else float("nan"),
    }


def group_tests(frame: pd.DataFrame, full: pd.DataFrame, *, seed: int = 7) -> dict:
    """What each group of player-level inputs is worth, on seasons the model never saw.

    A group that is in the model is taken out and the test is run again: the
    amount the score gets worse is what the group was adding. A group that
    was left out of the model is put in: the amount the score improves is what
    it would add. Either way a value above zero means "this group helps".
    """
    out = {}
    for name, columns in players.GROUPS.items():
        used = [c for c in columns if c in inputs.MODEL_FEATURES]
        if used:
            without = tuple(c for c in inputs.MODEL_FEATURES if c not in used)
            other = evaluate.walk_forward(frame, seed=seed, features=without)
            gap = log_loss_gap(other, full)
        else:
            other = evaluate.walk_forward(frame, seed=seed, features=inputs.MODEL_FEATURES + tuple(columns))
            gap = log_loss_gap(full, other)
        out[name] = {"in_model": bool(used), "columns": list(used or columns), "helps_by": gap}
        log.info("input group %s: worth %+.5f", name, gap["diff"])
    return out


def _previous_group_tests(save_to: Path | None) -> dict | None:
    """Group tests from the last saved run, if it used the same inputs."""
    if save_to is None or not save_to.exists():
        return None
    try:
        earlier = json.loads(save_to.read_text())
    except (OSError, ValueError):
        return None
    if earlier.get("inputs") == list(inputs.MODEL_FEATURES):
        return earlier.get("input_groups")
    return None


def run(*, seed: int = 7, save_to: Path | None = None, groups: bool = False) -> dict:
    """Run the whole test on the built data and return every result.

    ``groups`` also measures what each group of player-level inputs is worth.
    That repeats the season-by-season test once per group, so it is slow.
    """
    tables = build.load_all()
    odds = tables["odds"]
    frame = inputs.build_frame(tables)
    preds = evaluate.walk_forward(frame, seed=seed, log=log)
    team_only = evaluate.walk_forward(frame, seed=seed, features=inputs.TEAM_FEATURES)
    preds = preds.merge(team_only[["game_id", "p_model"]].rename(columns={"p_model": "p_team_only"}), on="game_id")
    forecast = evaluate.summarize(preds, odds)
    # The same test with lineups guessed from recent games, as on the morning of a game.
    # The model is still trained on real lineups, as it is in daily use.
    morning = evaluate.walk_forward(frame, seed=seed, test_frame=inputs.build_frame(tables, lineup_mode="projected"))
    forecast["overall"]["model_with_guessed_lineups"] = evaluate.score(
        morning["home_win"].to_numpy(dtype=float), morning["p_model"].to_numpy())

    priced = priced_predictions(preds, odds)
    priced_morning = priced_predictions(morning, odds)
    nothing = {"seasons": []}
    betting = betting_report(priced) if len(priced) else dict(nothing)
    # Price shopping is judged only on games that several sportsbooks quoted, at both kinds of price.
    shoppable = priced[priced["n_books"] >= MIN_BOOKS_TO_SHOP] if len(priced) else priced
    best = betting_report(shoppable, prices=BEST) if len(shoppable) else dict(nothing)
    best_middle = betting_report(shoppable, prices=CLOSE) if len(shoppable) else dict(nothing)
    early = betting_report(priced_morning, prices=OPEN, min_train_seasons=1) if len(priced) else dict(nothing)
    early["line_movement"] = line_movement(priced_morning) if len(priced) else {"games": 0}
    preds.merge(odds, on="game_id", how="left").to_parquet(
        ensure(processed_dir()) / "walk_forward.parquet", index=False)
    result = {
        "test_seasons": sorted(int(s) for s in preds["season"].unique()),
        "games_tested": int(len(preds)),
        "inputs": list(inputs.MODEL_FEATURES),
        "team_inputs": len(inputs.TEAM_FEATURES),
        "player_inputs": len(inputs.PLAYER_INPUTS),
        "forecast": forecast,
        "player_detail": {"model_minus_team_only_log_loss": log_loss_gap(preds, team_only)},
        "market_blend": {"closing": market_blend(priced) if len(priced) else {"games": 0},
                         "opening": market_blend(priced_morning, "open_market_home_prob", min_train_seasons=1)
                         if len(priced) else {"games": 0}},
        "betting": betting,
        "best_price": best,
        "best_price_middle": best_middle,
        "early_prices": early,
    }
    tested = group_tests(frame, preds, seed=seed) if groups else _previous_group_tests(save_to)
    if tested:
        result["input_groups"] = tested
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(result, indent=1, default=float))
    return result
