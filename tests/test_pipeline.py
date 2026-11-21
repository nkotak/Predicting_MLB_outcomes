"""The whole flow on a made-up league: score, backtest, train, predict, grade, learn."""

import json

import numpy as np
import pandas as pd
import pytest

from mlb_predict import backtest, build, cli, evaluate, inputs, players, predict
from mlb_predict.model import WinModel

from .conftest import make_odds


@pytest.fixture(scope="module")
def frame(big_league):
    return inputs.build_frame(big_league)


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


def test_model_has_three_parts_and_one_learns_from_the_run_margin(frame, preds):
    assert {"p_linear", "p_trees", "p_margin"} <= set(preds.columns)
    played = frame[frame["home_win"].notna()]
    train, calib = played[played["season"].isin([2013, 2014, 2015])], played[played["season"] == 2016]
    model = WinModel().fit(train, calib)
    parts = model.components(calib)
    assert set(parts) == {"linear", "trees", "margin"}
    assert all(((0 < p) & (p < 1)).all() for p in parts.values())
    # The margin part is taught the score. Change the scores without changing any winner and only it moves.
    wider = train.copy()
    home_won = wider["home_score"] > wider["away_score"]
    wider.loc[home_won, "home_score"] += 3
    wider.loc[~home_won, "away_score"] += 3
    again = WinModel().fit(wider, calib).components(calib)
    assert np.allclose(again["linear"], parts["linear"]) and np.allclose(again["trees"], parts["trees"])
    assert not np.allclose(again["margin"], parts["margin"])
    # The three parts mostly agree about which games are lopsided.
    assert np.corrcoef(parts["margin"], parts["linear"])[0, 1] > 0.8


def test_price_taught_part_learns_from_past_prices_and_never_sees_its_own(big_league, monkeypatch):
    from mlb_predict import model as model_module
    from mlb_predict.model import logit
    tables = dict(big_league)
    tables["odds"] = make_odds(big_league)
    priced_frame = inputs.build_frame(tables)
    assert inputs.TEACHER not in inputs.MODEL_FEATURES
    assert priced_frame.loc[priced_frame["status"] == "scheduled", inputs.TEACHER].isna().all()  # unplayed: no target
    played = priced_frame[priced_frame["home_win"].notna()]
    train, calib = played[played["season"].isin([2013, 2014, 2015])], played[played["season"] == 2016]
    test = played[played["season"] == 2017]
    # Too few priced games: the part is left out and the model is the three result-taught parts.
    without = WinModel().fit(train, calib)
    assert "prices" not in without.components(test) and without.info["price_taught_games"] == 0
    monkeypatch.setattr(model_module, "MIN_TEACHER_GAMES", 1000)
    taught = WinModel().fit(train, calib)
    parts = taught.components(test)
    assert "prices" in parts and taught.info["price_taught_games"] == int(train[inputs.TEACHER].notna().sum())
    assert not np.allclose(taught.predict(test), without.predict(test))
    # It follows the market it learned from.
    assert np.corrcoef(logit(parts["prices"]), logit(test[inputs.TEACHER]))[0, 1] > 0.6
    # A game's own price is never read: change every test game's price and no forecast moves.
    moved = test.assign(**{inputs.TEACHER: 0.9})
    assert np.allclose(taught.predict(moved), taught.predict(test))
    # In the season-by-season test the part is scored only on the seasons that have it.
    wf = evaluate.walk_forward(priced_frame)
    summary = evaluate.summarize(wf, tables["odds"])["overall"]
    assert wf["p_prices"].notna().any() and summary["price_taught_seasons"] == sorted(
        int(s) for s in wf.loc[wf["p_prices"].notna(), "season"].unique())
    assert summary["price_taught_only"]["games"] == summary["model_on_price_taught_seasons"]["games"]


def test_forecasts_can_be_made_on_rows_with_less_information(big_league, frame, preds):
    """Train on real lineups, forecast with lineups guessed that morning: the daily-use case."""
    guessed = inputs.build_frame(big_league, lineup_mode="projected")
    morning = evaluate.walk_forward(frame, test_frame=guessed)
    assert list(morning["game_id"]) == list(preds["game_id"])
    assert not np.allclose(morning["p_model"], preds["p_model"])
    assert (morning["p_model"] - preds["p_model"]).abs().mean() < 0.03  # close, because a guess is mostly right
    same = evaluate.walk_forward(frame, test_frame=frame)
    assert np.allclose(same["p_model"], preds["p_model"])


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
    assert info["features"] == list(inputs.MODEL_FEATURES) and info["scikit_learn"]
    model.features = ("something_else",)
    model.save(tmp_path, "stale")
    with pytest.raises(RuntimeError):
        WinModel.load(tmp_path, "stale")


def test_betting_report_compares_strategies_on_unseen_seasons(big_league, preds):
    odds = make_odds(big_league)
    priced = backtest.priced_predictions(preds, odds)
    report = backtest.betting_report(priced)
    assert report["seasons"] == [2020, 2021]  # three earlier priced seasons are needed to train the agent
    assert set(report["overall"]) == {"grpo_agent", "grpo_agent_safety_off", "edge_rule", "always_home",
                                      "always_favourite", "always_underdog"}
    for note in report["agent_training"].values():
        if note["kept_never_bet_policy"]:  # the guarded agent passes, whatever the raw policy would do
            assert note["bet_rate"] == 0.0 and note["bet_rate_safety_off"] >= 0.0
    assert report["agent_training"][2020]["train_seasons"] == [2017, 2018]
    assert report["agent_training"][2020]["validation_season"] == 2019
    # Blindly backing one kind of team at prices with a 4% cut loses money.
    assert report["overall"]["always_favourite"]["roi"] < 0.02
    assert report["overall"]["always_home"]["bets"] == report["games"]
    # At the closing price a bet cannot beat the close: it gives up about half the cut.
    assert report["overall"]["always_home"]["value_against_close"] == pytest.approx(-0.04 / 1.04, abs=1e-6)


def test_early_price_report_and_line_movement(big_league, preds):
    priced = backtest.priced_predictions(preds, make_odds(big_league))
    early = backtest.betting_report(priced, prices=backtest.OPEN, min_train_seasons=1)
    assert early["seasons"] == [2019, 2020, 2021]  # one training season is enough here
    move = backtest.line_movement(priced)
    assert move["games"] == len(priced) and move["log_loss_closing_price"] < move["log_loss_opening_price"]
    assert -1 < move["correlation"] < 1 and move["average_move_points"] > 0
    assert backtest.line_movement(priced.drop(columns=["open_home_dec"]))["games"] == 0


def test_best_price_never_pays_less_than_the_middle_price(big_league, preds):
    priced = backtest.priced_predictions(preds, make_odds(big_league))
    middle = backtest.betting_report(priced)
    best = backtest.betting_report(priced, prices=backtest.BEST)
    assert best["games"] == middle["games"]
    assert best["overall"]["always_home"]["roi"] > middle["overall"]["always_home"]["roi"]
    # A price better than the middle of the market beats the fair closing price by less of the cut.
    assert best["overall"]["always_home"]["value_against_close"] > middle["overall"]["always_home"]["value_against_close"]


def test_market_mix_reports_how_much_the_model_adds(big_league, preds):
    priced = backtest.priced_predictions(preds, make_odds(big_league))
    mix = backtest.market_blend(priced)
    assert mix["seasons"] == [2019, 2020, 2021] and mix["games"] == int(priced["season"].isin([2019, 2020, 2021]).sum())
    assert mix["log_loss_mix"] <= mix["log_loss_model"] + 0.002
    assert 0.0 <= mix["model_share"] <= 1.0 or mix["weight_on_model"] < 0
    assert mix["mix_minus_market"]["low"] <= mix["mix_minus_market"]["diff"] <= mix["mix_minus_market"]["high"]
    # A model that is the truth plus nothing else gets all the weight.
    perfect = priced.assign(p_model=np.clip(priced["home_win"] * 0.6 + 0.2, 0.2, 0.8))
    assert backtest.market_blend(perfect)["model_share"] > 0.8
    assert backtest.market_blend(priced.drop(columns=["market_home_prob"]))["games"] == 0


def test_player_detail_is_compared_with_the_team_only_model(frame, preds, monkeypatch):
    team_only = evaluate.walk_forward(frame, features=inputs.TEAM_FEATURES)
    gap = backtest.log_loss_gap(preds, team_only)
    assert gap["games"] == len(preds) and gap["low"] <= gap["diff"] <= gap["high"]
    assert backtest.log_loss_gap(preds, preds)["diff"] == 0.0
    # Group tests: one group in the model is taken out, one that is not in the model is put in.
    some = inputs.PLAYER_INPUTS[:3]
    monkeypatch.setattr(players, "GROUPS", {"first_three": tuple(some), "made_up": ("d_elo_copy",)})
    tested = backtest.group_tests(frame.assign(d_elo_copy=frame["d_elo"]), preds)
    assert tested["first_three"]["in_model"] and tested["first_three"]["columns"] == list(some)
    assert not tested["made_up"]["in_model"] and tested["made_up"]["helps_by"]["games"] == len(preds)


def test_value_against_close_by_hand():
    bets = pd.DataFrame({
        "side": ["home", "away", "pass"], "home_dec": [2.2, 1.8, 2.0], "away_dec": [1.7, 2.1, 1.9],
        "close_home_prob": [0.5, 0.55, 0.5],
    })
    # home at 2.2 when the fair close is 50%: +10%. away at 2.1 when the fair close is 45%: -5.5%.
    assert backtest.value_against_close(bets) == pytest.approx((2.2 * 0.5 - 1 + 2.1 * 0.45 - 1) / 2)
    assert np.isnan(backtest.value_against_close(bets[bets["side"] == "pass"]))


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
    assert result["inputs"] == list(inputs.MODEL_FEATURES) and "input_groups" not in result
    assert result["best_price"]["seasons"] == result["betting"]["seasons"] == result["best_price_middle"]["seasons"]
    assert result["market_blend"]["closing"]["games"] > 0
    assert "team_only" in result["forecast"]["overall"] and "model_with_guessed_lineups" in result["forecast"]["overall"]
    assert result["player_detail"]["model_minus_team_only_log_loss"]["games"] == result["games_tested"]
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
    assert "posted" in shown and "guessed" in shown and "once lineups are posted" in shown
    picks = pd.read_csv("picks.csv")
    assert len(picks) == 4 and picks["made_before_game"].all() and picks["p_home"].between(0.05, 0.95).all()
    assert picks["lineups_posted"].sum() == 3 and (picks["best_home_dec"] >= picks["home_dec"]).all()
    assert set(picks["agent_side"]) <= {"home", "away", "pass"}
    assert set(picks["agent_lean_side"]) <= {"home", "away", "pass"} and "Agent lean" in shown

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


def test_results_page_is_written_from_the_saved_backtest(big_league, preds, tmp_path, monkeypatch):
    from mlb_predict import report
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path / "data"))
    odds = make_odds(big_league)
    priced = backtest.priced_predictions(preds, odds)
    early = backtest.betting_report(priced, prices=backtest.OPEN, min_train_seasons=1)
    early["line_movement"] = backtest.line_movement(priced)
    result = {"test_seasons": sorted(int(s) for s in preds["season"].unique()), "games_tested": len(preds),
              "inputs": list(inputs.MODEL_FEATURES), "forecast": evaluate.summarize(preds, odds),
              "betting": backtest.betting_report(priced), "early_prices": early}
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "backtest.json").write_text(json.dumps(result, default=float))
    page = report.write_page(tmp_path / "reports" / "index.html").read_text()
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    for expected in ("How good are the forecasts?", "Against the betting market", "Paper betting at closing prices",
                     "Betting early, at opening prices", "copyrighted by Retrosheet", "<svg"):
        assert expected in page
    assert "—" not in page and "nan%" not in page
    markdown = report.backtest_markdown(json.loads((tmp_path / "reports" / "backtest.json").read_text()))
    assert "## Paper betting at closing prices" in markdown and "## Betting early, at opening prices" in markdown
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{report.README_START}\nold numbers\n{report.README_END}\n\n## Rest\n")
    assert report.update_readme(result, readme)
    text = readme.read_text()
    assert "old numbers" not in text and "## What the results say" in text and text.endswith("## Rest\n")
    assert "nan" not in text and "\u2014" not in text
    assert not report.update_readme(result, tmp_path / "missing.md")
    fragment = report.render_page(result, fragment=True)
    assert "<title>MLB Predict Results</title>" in fragment
    assert "<!doctype" not in fragment and "</html>" not in fragment and "<body" not in fragment
    with pytest.raises(FileNotFoundError):
        report.write_page(tmp_path / "elsewhere" / "index.html")
