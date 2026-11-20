"""Matching odds to games, and the checks that stop a bad build."""

import numpy as np
import pandas as pd
import pytest

from mlb_predict import build

from .conftest import make_league


def _games(rows):
    frame = pd.DataFrame(rows, columns=["game_id", "date", "home", "away", "number"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _events(rows):
    frame = pd.DataFrame(rows, columns=["event_id", "date", "start_utc", "home", "away", "home_dec"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    frame["away_dec"], frame["market_home_prob"], frame["n_books"] = 2.0, 0.5, 4
    frame["odds_fetched_utc"] = "2024-01-01T00:00:00Z"
    return frame


def test_doubleheader_prices_go_to_the_right_game():
    games = _games([("20240501-NYY-1", "2024-05-01", "NYY", "BOS", 1), ("20240501-NYY-2", "2024-05-01", "NYY", "BOS", 2),
                    ("20240501-LAD-0", "2024-05-01", "LAD", "SF", 0)])
    events = _events([("late", "2024-05-01", "2024-05-01T23:05Z", "NYY", "BOS", 1.7),
                      ("early", "2024-05-01", "2024-05-01T17:05Z", "NYY", "BOS", 2.1),
                      ("west", "2024-05-02", "2024-05-02T02:10Z", "LAD", "SF", 1.5)])
    events.loc[events["event_id"] == "west", "date"] = pd.Timestamp("2024-05-01")
    odds = build.match_odds(games, events).set_index("game_id")
    assert odds.loc["20240501-NYY-1", "event_id"] == "early" and odds.loc["20240501-NYY-1", "home_dec"] == 2.1
    assert odds.loc["20240501-NYY-2", "event_id"] == "late"
    assert odds.loc["20240501-LAD-0", "home_dec"] == 1.5


def test_a_game_listed_a_day_off_is_still_matched_once():
    games = _games([("20240502-CHC-0", "2024-05-02", "CHC", "STL", 0), ("20240503-CHC-0", "2024-05-03", "CHC", "STL", 0)])
    events = _events([("moved", "2024-05-01", "2024-05-01T23:05Z", "CHC", "STL", 1.9),
                      ("normal", "2024-05-03", "2024-05-03T23:05Z", "CHC", "STL", 1.8)])
    odds = build.match_odds(games, events).set_index("game_id")
    assert odds.loc["20240502-CHC-0", "event_id"] == "moved"
    assert odds.loc["20240503-CHC-0", "event_id"] == "normal"
    assert odds.index.is_unique


def test_events_without_a_price_or_a_game_are_dropped():
    games = _games([("20240501-NYY-0", "2024-05-01", "NYY", "BOS", 0)])
    events = _events([("priced", "2024-05-01", "2024-05-01T23:05Z", "NYY", "BOS", 1.7),
                      ("orphan", "2024-06-01", "2024-06-01T23:05Z", "HOU", "ATL", 1.7)])
    events.loc[len(events)] = ["no_price", pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-01T20:00Z"),
                               "LAD", "SF", np.nan, 2.0, 0.5, 4, None]
    assert list(build.match_odds(games, events)["event_id"]) == ["priced"]
    assert build.match_odds(games, events.iloc[0:0]).empty


@pytest.fixture()
def tables():
    made = make_league(seed=8, seasons=(2015,), days=20, scheduled=2)
    return {k: v.copy() for k, v in made.items()}


def _problems(tables):
    with pytest.raises(build.DataCheckError) as caught:
        build.check(tables)
    return str(caught.value)


def test_a_short_season_is_reported(tables):
    assert "outside the expected range" in _problems(tables)


def test_checks_catch_broken_data(tables):
    broken = dict(tables)
    broken["games"] = pd.concat([tables["games"], tables["games"].head(1)])
    assert "duplicate game ids" in _problems(broken)

    broken = dict(tables)
    broken["team_games"] = tables["team_games"].iloc[1:]
    assert "do not have exactly two team rows" in _problems(broken)

    broken = dict(tables)
    wrong = tables["team_games"].copy()
    wrong.loc[wrong.index[0], "runs"] += 3
    broken["team_games"] = wrong
    assert "runs do not match the final score" in _problems(broken)

    broken = dict(tables)
    extra = tables["pitcher_games"].copy()
    extra.loc[extra.index[1], "is_start"] = True
    broken["pitcher_games"] = extra
    assert "more than one starting pitcher" in _problems(broken)

    broken = dict(tables)
    unmatched = tables["games"].copy()
    unmatched["home_sp_id"] = "r:" + unmatched["home_sp_id"].astype(str)
    broken["games"] = unmatched
    assert "no MLB id in the crosswalk" in _problems(broken)

    broken = dict(tables)
    odd = tables["games"].copy()
    odd.loc[odd.index[0], "home"] = "XXX"
    broken["games"] = odd
    assert "unknown team codes" in _problems(broken)


def test_in_game_prices_are_caught(monkeypatch):
    """If favourites win far too often, the prices were taken during the game."""
    monkeypatch.setattr(build, "MIN_GAMES_FOR_MARKET_CHECK", 300)
    made = make_league(seed=4, seasons=(2015,), days=160, scheduled=0)
    games = made["games"]
    home_won = games["home_score"] > games["away_score"]
    made["odds"] = pd.DataFrame({
        "game_id": games["game_id"], "home_dec": np.where(home_won, 1.2, 5.0),
        "away_dec": np.where(home_won, 5.0, 1.2), "market_home_prob": np.where(home_won, 0.8, 0.2), "n_books": 3,
    })
    message = _problems(made)
    assert "in-game prices" in message


def test_save_and_load_round_trip(tables, tmp_path, monkeypatch):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path))
    build.save({k: tables[k] for k in ("games", "players")})
    again = build.load(("games", "players"))
    assert len(again["games"]) == len(tables["games"])
    with pytest.raises(FileNotFoundError) as caught:
        build.load(("games", "odds"))
    assert "mlb fetch" in str(caught.value)
