"""The whole flow on a made-up league: score, backtest, train, predict, grade, learn."""

import json

import numpy as np
import pandas as pd
import pytest

from mlb_predict import backtest, build, cli, evaluate, features, predict
from mlb_predict.model import WinModel

from .conftest import make_odds


@pytest.fixture(scope="module")
def frame(big_league):
    return features.build_features(big_league["games"], big_league["team_games"], big_league["pitcher_games"])


@pytest.fixture(scope="module")
def preds(frame):
    return evaluate.walk_forward(frame)


def test_scores_match_hand_calculation():
    s = evaluate.score([1, 0, 1, 1], [0.8, 0.3, 0.4, 0.6])
    assert s["games"] == 4 and s["accuracy"] == 0.75
    assert s["log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.7) + np.log(0.4) + np.log(0.6)) / 4)
    assert s["brier"] == pytest.approx((0.04 + 0.09 + 0.36 + 0.16) / 4)


def test_calibration_error_ignores_empty_ranges():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.40, 0.65, 20000)
    y = (rng.random(20000) < p).astype(float)
    # Well calibrated forecasts that only use part of the 0 to 1 range score close to zero.
    assert evaluate.calibration_error(y, p) < 0.02
    assert evaluate.calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.15


def test_walk_forward_never_trains_on_the_season_it_scores(frame, preds):
    assert sorted(preds["season"].unique()) == [2017, 2018, 2019, 2020, 2021]
    played = frame[frame["home_win"].notna()]
    assert len(preds) == len(played[played["season"] >= 2017])
    assert preds["p_model"].between(0.01, 0.99).all()
    model = WinModel().fit(played[played["season"].isin([2013, 2014, 2015])], played[played["season"] == 2016])
    assert model.info["train_seasons"] == [2013, 2014, 2015] and model.info["calibration_seasons"] == [2016]
    first = preds[preds["season"] == 2017].set_index("game_id")["p_model"]
    again = pd.Series(model.predict(played[played["season"] == 2017]),
                      index=played.loc[played["season"] == 2017, "game_id"])
    pd.testing.assert_series_equal(first.sort_index(), again.sort_index(), check_names=False)


def test_model_beats_the_simple_yardsticks_when_there_is_something_to_learn(preds):
    summary = evaluate.summarize(preds)["overall"]
    assert summary["model"]["log_loss"] < summary["always_home"]["log_loss"] - 0.01
    assert summary["model"]["accuracy"] > summary["always_home"]["accuracy"] + 0.03
    assert summary["model"]["calibration_error"] < 0.04
    assert summary["model_minus_always_home_log_loss"]["high"] < 0


def test_model_refuses_too_little_data_and_mismatched_inputs(frame, tmp_path):
    played = frame[frame["home_win"].notna()]
    with pytest.raises(ValueError):
        WinModel().fit(played.head(300), played.tail(300))
    model = WinModel().fit(played[played["season"] <= 2016], played[played["season"] == 2017])
    model.save(tmp_path)
    assert WinModel.load(tmp_path).predict(played.tail(5)) == pytest.approx(model.predict(played.tail(5)))
    info = json.loads((tmp_path / "win_model.json").read_text())
    assert info["features"] == list(features.FEATURES) and info["scikit_learn"]
    model.features = ("something_else",)
    model.save(tmp_path, "stale")
    with pytest.raises(RuntimeError):
        WinModel.load(tmp_path, "stale")


def test_betting_report_compares_strategies_on_unseen_seasons(big_league, preds):
    odds = make_odds(big_league)
    priced = backtest.priced_predictions(preds, odds)
    report = backtest.betting_report(priced)
    assert report["seasons"] == [2020, 2021]  # three earlier priced seasons are needed to train the agent
    assert set(report["overall"]) == {"grpo_agent", "edge_rule", "always_home", "always_favourite", "always_underdog"}
    assert report["agent_training"][2020]["train_seasons"] == [2017, 2018]
    assert report["agent_training"][2020]["validation_season"] == 2019
    # Blindly backing one kind of team at prices with a 4% cut loses money.
    assert report["overall"]["always_favourite"]["roi"] < 0.02
    assert report["overall"]["always_home"]["bets"] == report["games"]


def test_command_line_flow_from_build_to_scorecard(big_league, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MLB_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    tables = dict(big_league)
    tables["odds"] = make_odds(big_league)
    build.save(tables)

    assert cli.main(["backtest"]) == 0
    result = json.loads((tmp_path / "reports" / "backtest.json").read_text())
    assert result["test_seasons"] == [2017, 2018, 2019, 2020, 2021]
    assert "Paper betting" in (tmp_path / "reports" / "BACKTEST.md").read_text()

    assert cli.main(["train"]) == 0
    info = json.loads((tmp_path / "models" / "training_info.json").read_text())
    assert info["model"]["calibration_seasons"] == [2021] and info["agent"]["validation_season"] == 2021

    upcoming = big_league["games"][big_league["games"]["status"] == "scheduled"]
    day = upcoming["date"].iloc[0].date().isoformat()
    capsys.readouterr()
    assert cli.main(["predict", "--date", day, "--csv", "picks.csv"]) == 0
    shown = capsys.readouterr().out
    assert "paper trading only" in shown and "not announced" in shown
    picks = pd.read_csv("picks.csv")
    assert len(picks) == 4 and picks["made_before_game"].all() and picks["p_home"].between(0.05, 0.95).all()
    assert set(picks["agent_side"]) <= {"home", "away", "pass"}

    assert cli.main(["grade"]) == 0
    assert "Graded 0" in capsys.readouterr().out  # nothing has been played yet

    # The games are played: results arrive in the data.
    games = tables["games"].copy()
    rows = games["status"] == "scheduled"
    games.loc[rows, ["status", "home_score", "away_score"]] = ["final", 5.0, 3.0]
    build.save({"games": games})
    assert cli.main(["grade"]) == 0
    assert "Graded 4" in capsys.readouterr().out
    log = predict.read_log()
    assert log["graded"].all() and (log["home_win"] == 1.0).all()
    bet = log[log["rule_side"] == "home"]
    assert np.allclose(bet["rule_profit"], bet["home_dec"] - 1)  # profit per unit staked on a winner

    assert cli.main(["scorecard"]) == 0
    assert "Predictions made before the game" in capsys.readouterr().out
    card = predict.scorecard()
    assert card["made_before_game"]["model"]["games"] == 4 and card["replayed_after_game"] == {"games": 0}

    assert cli.main(["learn", "--rounds", "3"]) == 0
    history = json.loads((tmp_path / "models" / "agent_learning_history.json").read_text())
    assert len(history) == 1 and history[0]["games_learned_from"] > 1000


def test_helpful_errors_instead_of_tracebacks(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("MLB_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["predict", "--date", "2027-04-02"]) == 1
    assert "mlb fetch" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["predict", "--date", "April 2"])
