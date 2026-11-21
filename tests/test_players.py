"""Player-level inputs: no peeking, and each number means what its name says."""

import numpy as np
import pandas as pd
import pytest

from mlb_predict import inputs, players
from mlb_predict.features import order_key

from .conftest import make_league


def _detail(tables, **kwargs):
    return players.build_player_features(tables["games"], tables, **kwargs)


def _columns(frame):
    return [c for c in frame.columns if c != "game_id"]


def _order(tables):
    games = tables["games"]
    return pd.Series(order_key(pd.to_datetime(games["date"]), games["number"]).to_numpy(), index=games["game_id"])


def test_a_game_cannot_see_its_own_plate_appearances(league):
    """Rewrite everything that happened in one game. Its own inputs must not move."""
    before = _detail(league).set_index("game_id")
    games = league["games"]
    final = games[games["status"] == "final"].reset_index(drop=True)
    target = final.iloc[len(final) // 2]
    gid = target["game_id"]

    changed = {k: v.copy() for k, v in league.items()}
    g = changed["games"]
    row = g["game_id"] == gid
    g.loc[row, "home_score"], g.loc[row, "away_score"] = target["away_score"] + 9, target["home_score"]
    pa = changed["plate_appearances"]
    mine = pa["game_id"] == gid
    counted = [c for c in pa.columns if pa[c].dtype == float]
    pa.loc[mine, counted] = pa.loc[mine, counted] * 3 + 2

    after = _detail(changed).set_index("game_id")
    order = _order(league)
    same_or_earlier = order[order <= order[gid]].index
    pd.testing.assert_frame_equal(before.loc[same_or_earlier], after.loc[same_or_earlier])
    later = order[order > order[gid]].index
    moved = (before.loc[later] != after.loc[later]).any(axis=1)
    assert moved.sum() > 5, "later games should react, otherwise this test proves nothing"


def test_model_inputs_are_an_explicit_list_with_no_results_in_it(league):
    frame = inputs.build_frame(league).copy()
    frame["home_runs_today"] = frame["home_score"]  # a column that would give the answer away
    matrix = inputs.model_matrix(frame)
    assert tuple(matrix.columns) == inputs.MODEL_FEATURES
    assert set(inputs.PLAYER_INPUTS) <= set(players.PLAYER_FEATURES)
    banned = ("score", "home_win", "runs_today", "status")
    assert not [c for c in inputs.MODEL_FEATURES if any(word in c for word in banned)]
    assert len(frame) == len(league["games"]) and frame["game_id"].is_unique


def test_totals_count_earlier_games_only_and_older_seasons_for_less():
    events = pd.DataFrame({
        "who": [1, 1, 1, 1, 2],
        "season": [2019, 2020, 2021, 2021, 2021],
        "order": [201904010, 202004010, 202104010, 202104020, 202104010],
        "hits": [10.0, 20.0, 3.0, 4.0, 99.0],
    })
    queries = pd.DataFrame({"who": [1, 1, 1, 3], "season": [2021, 2021, 2022, 2021],
                            "order": [202104010, 202104020, 202204010, 202104020]})
    got = players.asof_totals(events, queries, ["who"], ["hits"])["hits"].tolist()
    # First 2021 game: nothing yet this season, last year at 0.6, two years ago at 0.35.
    assert got[0] == pytest.approx(0.6 * 20 + 0.35 * 10)
    # Second 2021 game adds the first one in full, and never its own 4 hits.
    assert got[1] == pytest.approx(3 + 0.6 * 20 + 0.35 * 10)
    # Next season: 2021 at 0.6, 2020 at 0.35, 2019 at the "older" weight of 0.2.
    assert got[2] == pytest.approx(0.6 * 7 + 0.35 * 20 + 0.2 * 10)
    assert got[3] == 0.0  # a player with no record
    career = players.asof_totals(events, queries, ["who"], ["hits"], career=True)["hits"].tolist()
    assert career[:3] == [30.0, 33.0, 37.0]


def test_recent_totals_cover_exactly_the_days_asked_for():
    events = pd.DataFrame({"who": [1, 1, 1, 1, 1, 2], "day": [10, 11, 20, 31, 32, 31],
                           "hits": [1.0, 2.0, 4.0, 8.0, 16.0, 99.0]})
    queries = pd.DataFrame({"who": [1, 1, 1, 3], "day": [32, 33, 12, 32]})
    got = players.window_totals(events, queries, ["who"], ["hits"], days=21)["hits"].tolist()
    assert got[0] == 2 + 4 + 8      # days 11 to 31: the game's own day (32) is never counted
    assert got[1] == 4 + 8 + 16     # days 12 to 32
    assert got[2] == 1 + 2          # days before 12
    assert got[3] == 0.0


def test_streaks_and_ages_stay_in_a_sensible_range(league):
    """Three weeks of results nudge a lineup's number by thousandths, never by tenths."""
    detail = _detail(league)
    assert 0 < detail["home_lu_hot"].abs().max() < 0.05
    assert 0 < detail["home_sp_hot"].abs().max() < 0.08
    assert detail["home_lu_age"].between(20, 45).all() and detail["home_sp_age"].between(20, 45).all()
    assert detail["home_sp_age"].nunique() > 5


def test_resting_the_best_hitter_lowers_the_lineup_number(league):
    """Swap a club's best hitter for a newcomer with no record: that game's numbers drop, nothing else moves."""
    base = _detail(league).set_index("game_id")
    games = league["games"]
    final = games[games["status"] == "final"].reset_index(drop=True)
    target = final.iloc[-40]
    gid, team = target["game_id"], target["home"]
    order = _order(league)
    pa = league["plate_appearances"]
    earlier = pa[pa["game_id"].map(order) < order[gid]]
    totals = earlier.groupby("batter")[["woba_num", "woba_den"]].sum()
    record = totals["woba_num"] / totals["woba_den"]
    lineup = league["lineups"]
    mine = lineup[(lineup["game_id"] == gid) & (lineup["team"] == team)]
    best = record[mine["player_id"]].idxmax()
    assert record[best] > totals["woba_num"].sum() / totals["woba_den"].sum()

    changed = {k: v.copy() for k, v in league.items()}
    swap = (changed["lineups"]["game_id"] == gid) & (changed["lineups"]["player_id"] == best)
    changed["lineups"].loc[swap, "player_id"] = "99999"
    after = _detail(changed).set_index("game_id")
    assert after.at[gid, "home_lu_woba"] < base.at[gid, "home_lu_woba"] - 0.0002
    assert after.at[gid, "home_lu_vs_usual"] < base.at[gid, "home_lu_vs_usual"]
    assert after.at[gid, "home_xr"] < base.at[gid, "home_xr"]
    assert after.at[gid, "away_lu_woba"] == pytest.approx(base.at[gid, "away_lu_woba"])
    earlier_games = order[order < order[gid]].index
    pd.testing.assert_frame_equal(base.loc[earlier_games], after.loc[earlier_games])


def test_a_lineup_that_is_not_posted_is_borrowed_from_recent_games(league):
    detail = _detail(league).set_index("game_id")
    games = league["games"]
    scheduled = games[games["status"] == "scheduled"]
    last = scheduled.iloc[-1]
    assert detail.at[last["game_id"], "home_lu_posted"] == 0.0 and detail.at[last["game_id"], "away_lu_posted"] == 0.0
    assert detail.at[last["game_id"], "home_lu_n"] == 9  # nine regulars from recent games
    assert np.isfinite(detail.loc[last["game_id"], list(players.PLAYER_FEATURES)].to_numpy(dtype=float)).all()
    posted = scheduled.iloc[0]
    assert detail.at[posted["game_id"], "home_lu_posted"] == 1.0

    projected = _detail(league, lineup_mode="projected").set_index("game_id")
    assert (projected["home_lu_posted"] == 0.0).all()
    final = games[games["status"] == "final"]
    late = final[final["season"] == final["season"].max()]["game_id"]
    # A guessed lineup is close to the real one but not identical.
    gap = (projected.loc[late, "home_lu_woba"] - detail.loc[late, "home_lu_woba"]).abs()
    assert 0 < gap.mean() < 0.01
    with pytest.raises(ValueError):
        _detail(league, lineup_mode="something else")


def test_a_guessed_lineup_leaves_out_players_who_left_the_roster(league):
    games = league["games"]
    last = games[games["status"] == "scheduled"].iloc[-1]
    gid, team = last["game_id"], last["home"]
    prepared = games.copy()
    prepared["order"] = order_key(pd.to_datetime(prepared["date"]), prepared["number"])
    prepared["home_sp"], prepared["away_sp"] = prepared["home_sp_id"], prepared["away_sp_id"]
    prepared["day"] = 0
    tg = players._team_games(prepared).assign(opp_hand="R")
    lineups = league["lineups"].rename(columns={"player_id": "batter"})
    rosters = league["rosters"].assign(pid=league["rosters"]["player_id"])
    guess = players.choose_lineups(tg, lineups, rosters)
    mine = guess[(guess["game_id"] == gid) & (guess["team"] == team)]
    assert len(mine) == 9 and sorted(mine["slot"]) == list(range(1, 10)) and (mine["posted"] == 0).all()
    # The guess starts from the club's latest real lineup.
    earlier = prepared[(prepared["status"] == "final") & ((prepared["home"] == team) | (prepared["away"] == team))]
    latest = earlier.sort_values("order")["game_id"].iloc[-1]
    latest_nine = set(lineups[(lineups["game_id"] == latest) & (lineups["team"] == team)]["batter"])
    assert set(mine["batter"]) == latest_nine
    # A regular who is no longer on the roster is replaced by someone who is.
    gone = mine["batter"].iloc[0]
    fewer = rosters[~((rosters["game_id"] == gid) & (rosters["pid"] == gone))]
    again = players.choose_lineups(tg, lineups, fewer)
    mine_again = again[(again["game_id"] == gid) & (again["team"] == team)]
    assert gone not in set(mine_again["batter"]) and len(mine_again) == 9
    # A posted lineup is always used as it is.
    posted_game = games[games["status"] == "final"].iloc[-1]["game_id"]
    shown = guess[guess["game_id"] == posted_game].sort_values(["team", "slot"])
    real = lineups[lineups["game_id"] == posted_game].sort_values(["team", "slot"])
    assert shown["batter"].tolist() == real["batter"].tolist() and (shown["posted"] == 1).all()


def test_ballpark_correction_uses_earlier_seasons_only(league):
    games = league["games"].copy()
    games["season"] = games["season"].astype("int64")
    pa = league["plate_appearances"].merge(games[["game_id", "season"]], on="game_id")
    factors = players.park_hitting_factors(pa, games, league["game_extras"])
    first = int(games["season"].min())
    assert first not in set(factors["season"])            # nothing earlier to measure from
    assert factors["woba_pf"].between(0.85, 1.15).all()
    # Inflate hitting at one park in the first season: later seasons' factor for it goes up, and
    # every plate appearance there in later seasons is scaled down.
    park = league["game_extras"]["venue"].iloc[0]
    there = pa["game_id"].isin(league["game_extras"].loc[league["game_extras"]["venue"] == park, "game_id"])
    boosted = pa.copy()
    boosted.loc[there & (boosted["season"] == first), "woba_num"] *= 1.5
    higher = players.park_hitting_factors(boosted, games, league["game_extras"]).set_index(["venue", "season"])["woba_pf"]
    before = factors.set_index(["venue", "season"])["woba_pf"]
    assert higher[(park, first + 1)] > before[(park, first + 1)]
    neutral = players._park_neutral(boosted, games, league["game_extras"])
    later = there & (boosted["season"] == first + 1) & (boosted["woba_num"] > 0)
    assert (neutral.loc[later, "woba_num"] < boosted.loc[later, "woba_num"]).all()
    same_season = there & (boosted["season"] == first)
    assert np.allclose(neutral.loc[same_season, "woba_num"], boosted.loc[same_season, "woba_num"])


def test_left_right_matchups_point_the_right_way(league):
    """In the made-up league a hitter does better against the opposite hand. The league table must say so."""
    games = league["games"].copy()
    games["date"] = pd.to_datetime(games["date"])
    games["season"] = games["season"].astype("int64")
    pa = league["plate_appearances"].merge(games[["game_id", "season"]], on="game_id")
    pa["bats"] = pa["batter"].map(league["people"].set_index("player_id")["bats"])
    pa["batter_is_pitcher"] = False
    table = players._league(pa, games, sorted(games["season"].unique()))
    shift = table["delta_b"].groupby(["bats", "pit_hand"])["delta_b"].mean()
    assert shift[("L", "L")] < 0 < shift[("L", "R")]
    assert shift[("R", "R")] < 0 < shift[("R", "L")]
    cell = table["cell"].groupby(["bat_side", "pit_hand"])["lg_cell"].mean()
    assert cell[("L", "R")] > cell[("L", "L")] and cell[("R", "L")] > cell[("R", "R")]
    # Each season's row is the season before it, so nothing from the season being predicted is used.
    first, second = sorted(games["season"].unique())[:2]
    rows = table["base"].set_index("season")
    assert rows.at[second, "lg_woba"] == rows.at[first, "lg_woba"]
    actual = pa[pa["season"] == first][["woba_num", "woba_den"]].sum()
    assert rows.at[second, "lg_woba"] == pytest.approx(actual["woba_num"] / actual["woba_den"])


def test_a_tired_bullpen_is_marked_down(league):
    detail = _detail(league).set_index("game_id")
    assert detail["home_bp_tired"].between(0.0, 1.0).all()
    games = league["games"].copy()
    games["date"] = pd.to_datetime(games["date"])
    final = games[games["status"] == "final"].sort_values("date")
    # Days off rest everyone: a club that has not played for three days has a fresh bullpen.
    rows = pd.concat([final[["game_id", "date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
                      final[["game_id", "date", "away"]].rename(columns={"away": "team"}).assign(side="away")])
    rows = rows.sort_values(["team", "date"])
    rows["gap"] = rows.groupby("team")["date"].diff().dt.days
    rested = rows[rows["gap"] >= 4]
    busy = rows[rows["gap"] == 1]
    assert len(rested) > 0 and len(busy) > 50
    tired = lambda part: np.mean([detail.at[g, f"{s}_bp_tired"] for g, s in zip(part["game_id"], part["side"])])
    assert tired(rested) == 0.0
    assert tired(busy) > 0.05


def test_injured_players_count_only_while_they_are_on_the_list(league):
    detail = _detail(league).set_index("game_id")
    games = league["games"].copy()
    games["date"] = pd.to_datetime(games["date"])
    spell = league["il_spells"].iloc[-1]  # a pitcher on the list in May
    home = games[(games["home"] == spell["team"]) & (games["date"].dt.year == spell["start"].year)]
    inside = home[(home["date"] >= spell["start"]) & (home["date"] <= spell["end"])]["game_id"]
    outside = home[(home["date"] > spell["end"] + pd.Timedelta(days=1))]["game_id"]
    assert len(inside) and len(outside)
    assert (detail.loc[inside, "home_il_n"] >= 1).all()
    assert (detail.loc[outside, "home_il_n"] == 0).all()
    without = {**league, "il_spells": league["il_spells"].iloc[0:0]}
    assert (_detail(without)["home_il_n"] == 0).all()


def test_statcast_numbers_come_from_the_season_before(league):
    base = _detail(league).set_index("game_id")
    games = league["games"]
    last_season = int(games["season"].max())
    changed = {k: v.copy() for k, v in league.items()}
    table = changed["sc_expected"]
    table.loc[table["season"] == last_season, "xwoba"] += 0.5   # this season's table must not be read
    after = _detail(changed).set_index("game_id")
    pd.testing.assert_frame_equal(base, after)
    table.loc[table["season"] == last_season - 1, "xwoba"] += 0.05
    moved = _detail(changed).set_index("game_id")
    this_year = games.loc[games["season"] == last_season, "game_id"]
    earlier = games.loc[games["season"] < last_season, "game_id"]
    assert (moved.loc[this_year, "home_lu_xgap"] > base.loc[this_year, "home_lu_xgap"]).all()
    pd.testing.assert_frame_equal(base.loc[earlier], moved.loc[earlier])


def test_every_player_input_is_filled_and_differs_between_games(league):
    detail = _detail(league)
    values = detail[list(players.PLAYER_FEATURES)]
    assert not values.isna().any().any()
    assert (values.std() > 0).all()
    grouped = [c for columns in players.GROUPS.values() for c in columns]
    assert sorted(grouped) == sorted(players.PLAYER_FEATURES), "every column belongs to exactly one group"


def test_wind_text_is_read_into_speed_and_direction():
    from mlb_predict.playerdata import parse_wind
    wind = parse_wind(pd.Series(["12 mph, Out To CF", "8 mph, In From LF", "5 mph, L To R", "", None]))
    assert wind["wind_mph"].tolist() == [12.0, 8.0, 5.0, 0.0, 0.0]
    assert wind["wind_out"].tolist() == [12.0, -8.0, 0.0, 0.0, 0.0]


def test_missing_optional_tables_do_not_break_anything():
    tables = make_league(seed=5, seasons=(2015, 2016), days=30)
    slim = {k: v for k, v in tables.items() if k not in ("il_spells", "sc_expected", "sc_fielding", "sc_speed")}
    detail = _detail(slim)
    assert (detail["d_il_n"] == 0).all() and (detail["d_lu_xgap"] == 0).all()
    # No roster listing at all: the bullpen falls back to whoever pitched in relief lately.
    no_rosters = {**slim, "rosters": slim["rosters"].iloc[0:0]}
    fallback = _detail(no_rosters)
    late = fallback.tail(40)
    assert late["home_bp_woba"].notna().all()
