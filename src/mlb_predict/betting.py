"""Betting maths and a paper-money simulator.

Paper trading only: this module works out what a bet would have returned. It
never places one.

Terms, in plain words:

* decimal odds: what a winning 1 unit bet pays back in total. Odds of 2.50
  return 2.50 (your 1 back plus 1.50 profit).
* fair chance: the win chance the price implies once the bookmaker's cut has
  been removed.
* edge: expected profit per unit staked, at the price actually on offer.
  ``edge = model chance x decimal odds - 1``. There is one definition of edge
  in this project and every decision and stake uses it.
* Kelly stake: the share of the bankroll that grows it fastest if the model's
  chance is right. Betting a fraction of it is the usual way to be careful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------------


def american_to_decimal(american: float) -> float:
    if american == 0 or -100 < american < 100:
        raise ValueError(f"{american} is not a valid American price")
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be above 1, got {decimal}")
    return (decimal - 1.0) * 100.0 if decimal >= 2.0 else -100.0 / (decimal - 1.0)


def implied_probability(decimal):
    """Win chance a price implies, bookmaker's cut still included."""
    return 1.0 / np.asarray(decimal, dtype=float)


def fair_probabilities(home_dec, away_dec):
    """Remove the bookmaker's cut from a two-way market.

    Returns the fair chance for the home side and for the away side. They add
    up to exactly 1.
    """
    home, away = implied_probability(home_dec), implied_probability(away_dec)
    total = home + away
    return home / total, away / total


def edge(prob, decimal):
    """Expected profit per unit staked at this price."""
    return np.asarray(prob, dtype=float) * np.asarray(decimal, dtype=float) - 1.0


def kelly_fraction(prob, decimal):
    """Full Kelly stake as a share of bankroll. Zero when there is no edge."""
    prob, decimal = np.asarray(prob, dtype=float), np.asarray(decimal, dtype=float)
    return np.clip(edge(prob, decimal) / (decimal - 1.0), 0.0, None)


# ---------------------------------------------------------------------------
# A simple rule-based strategy, used as the benchmark for the learning agent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdRule:
    """Bet the side with the better edge when that edge clears a bar.

    The stake is a fraction of the Kelly stake, capped at ``max_stake``.
    """

    min_edge: float = 0.03
    kelly_multiplier: float = 0.25
    max_stake: float = 0.02

    def decide(self, p_home, home_dec, away_dec) -> pd.DataFrame:
        p_home = np.asarray(p_home, dtype=float)
        home_dec, away_dec = np.asarray(home_dec, dtype=float), np.asarray(away_dec, dtype=float)
        edge_home, edge_away = edge(p_home, home_dec), edge(1.0 - p_home, away_dec)
        pick_home = edge_home >= edge_away
        best = np.where(pick_home, edge_home, edge_away)
        price = np.where(pick_home, home_dec, away_dec)
        prob = np.where(pick_home, p_home, 1.0 - p_home)
        stake = np.minimum(self.kelly_multiplier * kelly_fraction(prob, price), self.max_stake)
        bet = best >= self.min_edge
        return pd.DataFrame({
            "side": np.where(bet, np.where(pick_home, "home", "away"), "pass"),
            "stake": np.where(bet, stake, 0.0),
            "price": np.where(bet, price, np.nan),
            "edge": best,
        })


# ---------------------------------------------------------------------------
# Paper-money simulation
# ---------------------------------------------------------------------------


def settle(side, stake, home_dec, away_dec, home_win) -> np.ndarray:
    """Profit of each bet as a share of the bankroll it was staked from."""
    side = np.asarray(side)
    stake = np.asarray(stake, dtype=float)
    home_win = np.asarray(home_win, dtype=float)
    profit = np.zeros(len(side))
    for name, price, won in (("home", home_dec, home_win == 1), ("away", away_dec, home_win == 0)):
        mask = side == name
        price = np.asarray(price, dtype=float)
        profit[mask] = np.where(won[mask], stake[mask] * (price[mask] - 1.0), -stake[mask])
    return profit


def simulate(bets: pd.DataFrame, *, start_bankroll: float = 1000.0) -> dict:
    """Run a bankroll through a list of decisions, one day at a time.

    ``bets`` needs the columns ``date``, ``side`` (home, away or pass),
    ``stake`` (share of bankroll), ``home_dec``, ``away_dec`` and ``home_win``.

    All of a day's stakes are sized from the bankroll at the start of that
    day and settled together at the end of it, because games on one day
    overlap and winnings from one cannot fund another.
    """
    bets = bets.sort_values("date").reset_index(drop=True)
    placed = bets[bets["side"] != "pass"]
    result = {
        "games_offered": int(len(bets)),
        "bets": int(len(placed)),
        "start_bankroll": float(start_bankroll),
    }
    if placed.empty:
        result.update({"wins": 0, "win_rate": float("nan"), "staked_units": 0.0, "profit_units": 0.0,
                       "roi": float("nan"), "roi_low": float("nan"), "roi_high": float("nan"),
                       "final_bankroll": float(start_bankroll), "growth": 0.0, "max_drawdown": 0.0,
                       "bankroll_by_day": []})
        return result

    share = settle(placed["side"], placed["stake"], placed["home_dec"], placed["away_dec"], placed["home_win"])
    per_unit = share / placed["stake"].to_numpy()  # profit per 1 unit staked: the flat-stake view
    daily = pd.Series(share, index=placed["date"].to_numpy()).groupby(level=0).sum().sort_index()
    daily_staked = placed.groupby("date")["stake"].sum().sort_index()
    if (daily_staked > 1.0).any():
        raise ValueError("stakes on one day add up to more than the whole bankroll")

    bankroll = start_bankroll * np.cumprod(1.0 + daily.to_numpy())
    path = np.concatenate([[start_bankroll], bankroll])
    peak = np.maximum.accumulate(path)
    start_of_day = path[:-1]
    stake_by_day = dict(zip(daily.index, start_of_day))
    money_staked = (placed["stake"].to_numpy() * placed["date"].map(stake_by_day).to_numpy()).sum()

    rng = np.random.default_rng(5)
    draws = per_unit[rng.integers(0, len(per_unit), size=(2000, len(per_unit)))].mean(axis=1)
    result.update({
        "wins": int((share > 0).sum()),
        "win_rate": float((share > 0).mean()),
        "average_price": float(np.where(placed["side"] == "home", placed["home_dec"], placed["away_dec"]).mean()),
        "staked_units": float(money_staked),
        "profit_units": float(path[-1] - start_bankroll),
        # Return per unit staked if every bet were the same size. This is the
        # fairest single number for "does the selection make money".
        "roi": float(per_unit.mean()),
        "roi_low": float(np.quantile(draws, 0.025)),
        "roi_high": float(np.quantile(draws, 0.975)),
        "final_bankroll": float(path[-1]),
        "growth": float(path[-1] / start_bankroll - 1.0),
        "max_drawdown": float(((peak - path) / peak).max()),
        "bankroll_by_day": [(pd.Timestamp(d).strftime("%Y-%m-%d"), round(float(b), 2))
                            for d, b in zip(daily.index, bankroll)],
    })
    return result
