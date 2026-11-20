"""Betting maths with answers worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from mlb_predict import betting as bt


def test_american_and_decimal_prices_convert_both_ways():
    assert bt.american_to_decimal(150) == pytest.approx(2.5)
    assert bt.american_to_decimal(-200) == pytest.approx(1.5)
    assert bt.american_to_decimal(100) == pytest.approx(2.0)
    assert bt.decimal_to_american(2.5) == pytest.approx(150)
    assert bt.decimal_to_american(1.5) == pytest.approx(-200)
    for price in (-350, -110, 100, 105, 240):
        assert bt.decimal_to_american(bt.american_to_decimal(price)) == pytest.approx(price)


def test_invalid_prices_are_rejected_instead_of_crashing_later():
    with pytest.raises(ValueError):
        bt.american_to_decimal(0)
    with pytest.raises(ValueError):
        bt.american_to_decimal(50)
    with pytest.raises(ValueError):
        bt.decimal_to_american(1.0)


def test_removing_the_bookmakers_cut():
    home, away = bt.fair_probabilities(bt.american_to_decimal(-110), bt.american_to_decimal(-110))
    assert home == pytest.approx(0.5) and away == pytest.approx(0.5)
    home, away = bt.fair_probabilities(1.62, 2.32)  # about -160 / +132
    assert home + away == pytest.approx(1.0)
    assert home == pytest.approx((1 / 1.62) / (1 / 1.62 + 1 / 2.32))


def test_edge_and_kelly():
    assert bt.edge(0.55, 2.0) == pytest.approx(0.10)
    assert bt.kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert bt.kelly_fraction(0.60, 2.0) == pytest.approx(0.20)
    assert bt.kelly_fraction(0.45, 2.0) == 0.0  # no edge, no bet
    # A 55% winner at the standard -110 price returns 5% per unit, not 0%.
    assert bt.edge(0.55, bt.american_to_decimal(-110)) == pytest.approx(0.05)


def test_rule_uses_one_definition_of_edge_for_the_decision_and_the_stake():
    rule = bt.ThresholdRule(min_edge=0.03, kelly_multiplier=0.25, max_stake=0.02)
    out = rule.decide([0.56, 0.50, 0.40], [2.0, 1.90, 2.0], [1.90, 1.95, 1.75])
    # game 1: home edge 0.12, quarter Kelly = 0.03, capped at 0.02
    assert out.loc[0, "side"] == "home" and out.loc[0, "stake"] == pytest.approx(0.02)
    # game 2: both sides lose money, so pass with a zero stake
    assert out.loc[1, "side"] == "pass" and out.loc[1, "stake"] == 0.0
    # game 3: away edge 0.60 * 1.75 - 1 = 0.05, quarter Kelly = 0.25 * 0.05 / 0.75
    assert out.loc[2, "side"] == "away" and out.loc[2, "stake"] == pytest.approx(0.25 * 0.05 / 0.75)
    # Every bet that is chosen gets a stake above zero.
    assert (out.loc[out["side"] != "pass", "stake"] > 0).all()


def _bets(rows):
    return pd.DataFrame(rows, columns=["date", "side", "stake", "home_dec", "away_dec", "home_win"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_bankroll_is_settled_one_day_at_a_time():
    bets = _bets([
        ("2024-04-01", "home", 0.10, 2.0, 1.9, 1.0),   # wins 10% of the bankroll
        ("2024-04-01", "away", 0.10, 1.8, 2.5, 1.0),   # loses 10% of the same starting bankroll
        ("2024-04-02", "home", 0.05, 3.0, 1.4, 1.0),   # wins 10% of the new bankroll
        ("2024-04-02", "pass", 0.00, 2.0, 1.9, 0.0),
    ])
    run = bt.simulate(bets, start_bankroll=1000.0)
    assert run["bets"] == 3 and run["wins"] == 2 and run["games_offered"] == 4
    # Day 1 nets to zero because both stakes come from the 1,000 start. Day 2: +10%.
    assert run["final_bankroll"] == pytest.approx(1100.0)
    assert run["bankroll_by_day"] == [("2024-04-01", 1000.0), ("2024-04-02", 1100.0)]
    # Flat-stake view: +1.0, -1.0 and +2.0 per unit over three bets.
    assert run["roi"] == pytest.approx(2.0 / 3.0)
    assert run["staked_units"] == pytest.approx(100 + 100 + 50)


def test_drawdown_and_no_bets():
    losing = _bets([("2024-04-0%d" % d, "home", 0.10, 2.0, 1.9, 0.0) for d in range(1, 4)])
    run = bt.simulate(losing)
    assert run["final_bankroll"] == pytest.approx(1000 * 0.9 ** 3)
    assert run["max_drawdown"] == pytest.approx(1 - 0.9 ** 3)
    assert run["roi"] == pytest.approx(-1.0)
    idle = bt.simulate(_bets([("2024-04-01", "pass", 0.0, 2.0, 1.9, 1.0)]))
    assert idle["bets"] == 0 and idle["final_bankroll"] == 1000.0 and np.isnan(idle["roi"])


def test_staking_more_than_the_bankroll_in_a_day_is_an_error():
    bets = _bets([("2024-04-01", "home", 0.6, 2.0, 1.9, 1.0), ("2024-04-01", "away", 0.6, 2.0, 1.9, 1.0)])
    with pytest.raises(ValueError):
        bt.simulate(bets)


def test_backing_every_favourite_at_fair_prices_plus_a_cut_loses_about_the_cut():
    rng = np.random.default_rng(0)
    n = 40000
    truth = rng.uniform(0.35, 0.65, n)
    home_dec, away_dec = 1 / (truth * 1.04), 1 / ((1 - truth) * 1.04)
    bets = pd.DataFrame({
        "date": pd.Timestamp("2024-04-01") + pd.to_timedelta(np.arange(n) // 15, unit="D"),
        "side": np.where(truth >= 0.5, "home", "away"), "stake": 0.001,
        "home_dec": home_dec, "away_dec": away_dec, "home_win": (rng.random(n) < truth).astype(float),
    })
    run = bt.simulate(bets)
    assert run["roi"] == pytest.approx(-0.038, abs=0.012)
    assert run["roi_low"] < run["roi"] < run["roi_high"]
