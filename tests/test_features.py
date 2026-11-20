"""Model inputs: no peeking at the result, and each number means what its name says."""

import numpy as np
import pandas as pd
import pytest

from mlb_predict import features as ft
from mlb_predict.features import FEATURES, build_features, model_matrix

from .conftest import make_league


def _features(tables):
    return build_features(tables["games"], tables["team_games"], tables["pitcher_games"])


def _columns(frame):
    return [c for c in frame.columns if c.startswith(("home_", "away_", "d_")) and c not in
            ("home_score", "away_score", "home_win", "home_sp_id", "away_sp_id")] + list(ft.CONTEXT)


def test_a_game_cannot_see_its_own_result(league):
    """Rewrite one game's result and box score. Its own inputs must not move."""
    before = _features(league).set_index("game_id")
    games = league["games"]
    final = games[games["status"] == "final"].reset_index(drop=True)
    target = final.iloc[len(final) // 2]
    gid = target["game_id"]

    changed = {k: v.copy() for k, v in league.items()}
    g = changed["games"]
    row = g["game_id"] == gid
    g.loc[row, "home_score"], g.loc[row, "away_score"] = target["away_score"] + 9, target["home_score"]
    tg = changed["team_games"]
    numeric = [c for c in tg.columns if c not in ("game_id", "team", "is_home")]
    tg.loc[tg["game_id"] == gid, numeric] = tg.loc[tg["game_id"] == gid, numeric] * 3 + 5
    home_row = (tg["game_id"] == gid) & tg["is_home"]
    tg.loc[home_row, "runs"] = target["away_score"] + 9
    tg.loc[(tg["game_id"] == gid) & ~tg["is_home"], "runs"] = target["home_score"]
    pg = changed["pitcher_games"]
    stats = ["outs", "bf", "h", "hr", "r", "er", "bb", "so"]
    pg.loc[pg["game_id"] == gid, stats] = pg.loc[pg["game_id"] == gid, stats] * 2 + 7

    after = _features(changed).set_index("game_id")
    columns = _columns(before)
    order = before["order"]
    same_or_earlier = order[order <= order[gid]].index
    pd.testing.assert_frame_equal(before.loc[same_or_earlier, columns], after.loc[same_or_earlier, columns])
    later = order[order > order[gid]].index
    moved = (before.loc[later, columns] != after.loc[later, columns]).any(axis=1)
    assert moved.sum() > 5, "later games should react, otherwise this test proves nothing"


def test_model_only_sees_the_allowed_columns(league):
    frame = _features(league)
    frame["home_runs_today"] = frame["home_score"]  # a column that would give the answer away
    matrix = model_matrix(frame)
    assert tuple(matrix.columns) == FEATURES
    banned = ("score", "home_win", "runs_today", "status")
    assert not [c for c in FEATURES if any(word in c for word in banned)]
    assert not matrix.isna().any().any()


def test_team_record_counts_home_and_road_games(league):
    frame = _features(league)
    games = league["games"]
    final = games[games["status"] == "final"]
    target = frame[(frame["season"] == 2016) & (frame["status"] == "final")].iloc[150]
    team = target["home"]
    earlier = final[(final["season"] == 2016) & (final["date"] < target["date"])
                    & ((final["home"] == team) | (final["away"] == team))]
    assert target["home_gp"] == len(earlier)
    assert (earlier["away"] == team).any() and (earlier["home"] == team).any()
    wins = (((earlier["home"] == team) & (earlier["home_score"] > earlier["away_score"]))
            | ((earlier["away"] == team) & (earlier["away_score"] > earlier["home_score"]))).sum()
    last = final[(final["season"] == 2015) & ((final["home"] == team) | (final["away"] == team))]
    last_wins = (((last["home"] == team) & (last["home_score"] > last["away_score"]))
                 | ((last["away"] == team) & (last["away_score"] > last["home_score"]))).sum()
    prior = 0.5 * last_wins / len(last) + 0.5 * 0.5
    expected = (wins + ft.K_TEAM_GAMES * prior) / (len(earlier) + ft.K_TEAM_GAMES)
    assert target["home_win_pct"] == pytest.approx(expected, abs=1e-9)


def test_starter_numbers_are_his_own(league):
    """Strikeout rate comes from the pitcher's own earlier lines, pulled toward the league rate."""
    frame = _features(league)
    games, lines = league["games"], league["pitcher_games"]
    target = frame[(frame["season"] == 2017) & (frame["status"] == "final")].iloc[100]
    pitcher = target["home_sp_id"]
    when = lines.merge(games[["game_id", "date", "season"]], on="game_id")
    mine = when[(when["player_id"] == pitcher) & (when["date"] < target["date"])]
    weight = mine["season"].map({2017: 1.0, 2016: ft.SEASON_WEIGHTS[1], 2015: ft.SEASON_WEIGHTS[2]})
    so, bf = (mine["so"] * weight).sum(), (mine["bf"] * weight).sum()
    league_2016 = when[when["season"] == 2016]
    league_rate = league_2016["so"].sum() / league_2016["bf"].sum()
    expected = (so + ft.K_SP["k"] * league_rate) / (bf + ft.K_SP["k"])
    assert target["home_sp_k"] == pytest.approx(expected, rel=1e-9)
    assert target["home_sp_known"] == 1.0


def test_second_game_of_a_doubleheader_sees_the_first(league):
    frame = _features(league)
    second = frame[(frame["number"] == 2) & (frame["season"] == 2015)].iloc[0]
    first = frame[(frame["number"] == 1) & (frame["date"] == second["date"]) & (frame["home"] == second["home"])].iloc[0]
    assert second["home_gp"] == first["home_gp"] + 1
    assert second["home_rest"] == 0.0
    assert second["is_doubleheader"] == 1.0


def test_recent_form_restarts_each_season(league):
    frame = _features(league)
    opener = frame[frame["season"] == 2016].iloc[0]
    assert opener["home_gp"] == 0
    assert opener["home_form10"] == pytest.approx(0.5)
    assert opener["home_rd30"] == 0.0
    assert opener["home_rest"] == 5.0


def test_scheduled_games_get_inputs_and_unknown_starters_are_flagged(league):
    frame = _features(league)
    scheduled = frame[frame["status"] == "scheduled"]
    assert len(scheduled) == 4
    assert scheduled["home_win"].isna().all()
    assert not model_matrix(scheduled).isna().any().any()
    unknown = scheduled[scheduled["home_sp_id"].isna()].iloc[0]
    assert unknown["home_sp_known"] == 0.0
    assert unknown["away_sp_known"] == 1.0
    lines = league["pitcher_games"].merge(league["games"][["game_id", "season"]], on="game_id")
    # An unannounced starter is given last season's league-wide rate.
    last = lines[lines["season"] == unknown["season"] - 1]
    assert unknown["home_sp_k"] == pytest.approx(last["so"].sum() / last["bf"].sum())


def test_team_rating_moves_toward_the_winner_and_ignores_its_own_game():
    tables = make_league(seed=5, seasons=(2015,), days=3, scheduled=0)
    games = tables["games"].copy()
    games["order"] = ft.order_key(games["date"], games["number"])
    ratings = ft.elo_ratings(games).set_index("game_id")
    first = games.sort_values("order").iloc[0]
    assert ratings.loc[first["game_id"], "home_elo"] == ft.ELO_START
    later = games[(games["order"] > first["order"]) & ((games["home"] == first["home"]) | (games["away"] == first["home"]))]
    nxt = later.sort_values("order").iloc[0]
    rating = ratings.loc[nxt["game_id"], "home_elo" if nxt["home"] == first["home"] else "away_elo"]
    assert (rating > ft.ELO_START) == (first["home_score"] > first["away_score"])


def test_bullpen_workload_counts_only_the_last_three_days(league):
    frame = _features(league)
    games, lines = league["games"], league["pitcher_games"]
    target = frame[(frame["season"] == 2016) & (frame["status"] == "final")].iloc[200]
    team = target["home"]
    when = lines.merge(games[["game_id", "date"]], on="game_id")
    window = when[(when["team"] == team) & ~when["is_start"]
                  & (when["date"] < target["date"]) & (when["date"] >= target["date"] - pd.Timedelta(days=3))]
    assert target["home_pen_outs3"] == window["outs"].sum()


def test_every_input_differs_between_games(league):
    matrix = model_matrix(_features(league))
    constant = [c for c in matrix.columns if matrix[c].nunique() <= 1]
    assert not constant, f"inputs that never change: {constant}"
    assert np.isfinite(matrix.to_numpy()).all()
