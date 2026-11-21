"""The player-level downloads: game feed, injured list and Statcast tables. No network."""

import json

import pandas as pd
import pytest

from mlb_predict import playerdata
from mlb_predict.sources import gamefeed, mlbapi, savant, transactions

from .conftest import make_league


def _pitch(call, kind="FF", speed=95.0, hit=None):
    event = {"isPitch": True, "details": {"call": {"code": call}, "type": {"code": kind}},
             "pitchData": {"startSpeed": speed}}
    if hit:
        event["hitData"] = hit
    return event


def _play(half, inning, batter, pitcher, side, hand, event_type, pitches):
    return {"result": {"type": "atBat", "eventType": event_type},
            "about": {"halfInning": half, "inning": inning},
            "matchup": {"batter": {"id": batter}, "pitcher": {"id": pitcher},
                        "batSide": {"code": side}, "pitchHand": {"code": hand}},
            "playEvents": [{"isPitch": False, "details": {}}] + pitches}


def _side(team, first_id, pitchers, bullpen, *, finished=True):
    players = {}
    for slot in range(1, 10):
        pid = first_id + slot
        players[f"ID{pid}"] = {"person": {"id": pid}, "position": {"abbreviation": "PH" if slot == 9 else "1B"},
                               "allPositions": [{"abbreviation": "SS" if slot == 1 else "1B"}]}
        if finished:
            players[f"ID{pid}"]["battingOrder"] = f"{slot}00"
    sub = first_id + 50  # came in for the ninth hitter late in the game
    order = [first_id + slot for slot in range(1, 10)]
    batters = list(order)
    if finished:
        players[f"ID{sub}"] = {"person": {"id": sub}, "battingOrder": "901", "position": {"abbreviation": "PH"}}
        batters.append(sub)
        order = order[:8] + [sub]  # after a game the feed lists whoever finished in each slot
    return {"team": {"id": team}, "players": players, "battingOrder": order, "batters": batters,
            "pitchers": pitchers, "bullpen": bullpen, "bench": [first_id + 60]}


def _payload(finished=True):
    plays = [
        _play("top", 1, 201, 900, "L", "R", "single",
              [_pitch("B"), _pitch("C", "SL", 85.0), _pitch("D", hit={"launchSpeed": 101.5, "launchAngle": 12.0,
                                                                     "trajectory": "line_drive"})]),
        _play("top", 1, 202, 900, "R", "R", "strikeout", [_pitch("S"), _pitch("F"), _pitch("W", "CU", 80.0)]),
        _play("top", 1, 203, 900, "R", "R", "caught_stealing_2b", [_pitch("B")]),
        _play("bottom", 1, 101, 950, "S", "L", "walk", [_pitch("B", "SI", 93.0)] * 4),
        _play("bottom", 1, 102, 951, "R", "R", "home_run", [_pitch("E", "FC", 90.0, {"launchSpeed": 108.0,
                                                                                    "launchAngle": 28.0,
                                                                                    "trajectory": "fly_ball"})]),
    ] if finished else []
    return {
        "gamePk": 777,
        "gameData": {
            "datetime": {"officialDate": "2026-06-15"}, "status": {"detailedState": "Final" if finished else "Pre-Game"},
            "venue": {"id": 3313}, "weather": {"condition": "Clear", "temp": "81", "wind": "9 mph, Out To CF"},
            "teams": {"home": {"id": 147}, "away": {"id": 111}},
            "players": {"ID101": {"id": 101, "fullName": "Home One", "batSide": {"code": "S"},
                                  "pitchHand": {"code": "R"}, "birthDate": "1995-02-03",
                                  "primaryPosition": {"abbreviation": "SS"}}},
        },
        "liveData": {
            "boxscore": {
                "teams": {"home": _side(147, 100, [900, 901] if finished else [], [902, 903], finished=finished),
                          "away": _side(111, 200, [950, 951] if finished else [], [952], finished=finished)},
                "officials": [{"official": {"id": 42}, "officialType": "First Base"},
                              {"official": {"id": 43}, "officialType": "Home Plate"}],
            },
            "plays": {"allPlays": plays},
        },
    }


def test_game_feed_is_reduced_to_lineups_plays_and_setting():
    record = gamefeed.slim(_payload())
    assert (record["pk"], record["date"], record["venue"], record["temp"], record["umpire"]) == (
        777, "2026-06-15", 3313, 81, 43)
    home = record["sides"]["home"]
    # The starting nine, not whoever finished the game in each slot.
    assert [p for p, _ in home["lineup"]] == [101, 102, 103, 104, 105, 106, 107, 108, 109]
    assert home["lineup"][0][1] == "SS" and home["lineup"][8][1] == "1B"  # the position he started at
    assert 150 in home["batters"] and home["pitchers"] == [900, 901] and home["bench"] == [160]
    assert record["people"]["101"] == ["Home One", "S", "R", "1995-02-03", "SS"]
    single, strikeout, not_a_pa, walk, homer = record["plays"]
    fields = dict(zip(gamefeed.PA_FIELDS, single))
    assert fields["half"] == 0 and fields["event"] == gamefeed.EVENTS["single"] and fields["pitches"] == 3
    assert (fields["called"], fields["balls"], fields["in_play"], fields["whiffs"]) == (1, 1, 1, 0)
    assert fields["fb_n"] == 2 and fields["fb_speed_sum"] == 190.0  # the slider is not a fastball
    assert fields["launch_speed"] == 101.5 and fields["trajectory"] == "L"
    assert dict(zip(gamefeed.PA_FIELDS, strikeout))["whiffs"] == 2
    assert dict(zip(gamefeed.PA_FIELDS, not_a_pa))["event"] == 0
    assert dict(zip(gamefeed.PA_FIELDS, walk))["half"] == 1 and dict(zip(gamefeed.PA_FIELDS, homer))["fb_n"] == 0


def test_feed_tables_count_plate_appearances_correctly():
    tables = gamefeed.tables_from([gamefeed.slim(_payload())])
    lineups = tables["lineups"]
    assert len(lineups) == 18 and lineups.groupby("team_id")["slot"].apply(list).iloc[0] == list(range(1, 10))
    roles = tables["rosters"].groupby(["team_id", "role"]).size()
    assert roles[(147, "SP")] == 1 and roles[(147, "RP")] == 1 and roles[(147, "BP")] == 2 and roles[(147, "BN")] == 1
    assert roles[(147, "BAT")] == 10  # nine starters and the substitute
    pa = tables["plate_appearances"].set_index(["batter", "pitcher"])
    single = pa.loc[(201, 900)]
    assert single["pa"] == 1 and single["h1"] == 1 and single["woba_num"] == pytest.approx(0.88)
    assert single["bat_team_id"] == 111 and single["pit_team_id"] == 147 and single["vs_starter"]
    assert single["bip"] == 1 and single["bip_hit"] == 1 and single["hard"] == 1 and single["ev_n"] == 1
    assert pa.loc[(202, 900), "so"] == 1 and pa.loc[(202, 900), "swings"] == 3
    # A runner thrown out is not a plate appearance, but the pitch thrown still counts for the pitcher.
    assert pa.loc[(203, 900), "pa"] == 0 and pa.loc[(203, 900), "pitches"] == 1
    walk = pa.loc[(101, 950)]
    assert walk["bb"] == 1 and walk["ab"] == 0 and walk["woba_num"] == pytest.approx(0.69) and walk["woba_den"] == 1
    homer = pa.loc[(102, 951)]
    assert homer["hr"] == 1 and homer["bip"] == 0 and not homer["vs_starter"]  # a reliever gave it up
    extras = tables["game_extras"].iloc[0]
    assert extras["wind"] == "9 mph, Out To CF" and extras["umpire"] == 43
    assert tables["people"].iloc[0]["bats"] == "S"


def test_before_a_game_the_posted_lineup_is_used():
    record = gamefeed.slim(_payload(finished=False))
    assert [p for p, _ in record["sides"]["away"]["lineup"]] == list(range(201, 210))
    assert record["plays"] == [] and record["state"] == "Pre-Game"
    empty = _payload(finished=False)
    empty["liveData"]["boxscore"]["teams"]["home"]["battingOrder"] = []
    assert gamefeed.slim(empty)["sides"]["home"]["lineup"] == []  # not posted yet


def test_feed_downloads_are_stored_once_and_failures_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path))
    schedule = [{"game_pk": pk, "state": state} for pk, state in ((1, "Final"), (2, "Final"), (3, "Scheduled"))]
    monkeypatch.setattr(mlbapi, "fetch_schedule", lambda season, **_: schedule)
    calls = []

    def fake(url):
        pk = int(url.split("/game/")[1].split("/")[0])
        calls.append(pk)
        if pk == 2 and calls.count(2) == 1:
            raise gamefeed.FetchError("temporary")
        return {**_payload(), "gamePk": pk}

    monkeypatch.setattr(gamefeed, "get_json", fake)
    first = gamefeed.fetch(2026, workers=1)
    assert (first["downloaded"], first["failed"]) == (1, 1) and gamefeed.stored_pks(2026) == {1}
    second = gamefeed.fetch(2026, workers=1)
    assert (second["stored_before"], second["downloaded"], second["failed"]) == (1, 1, 0)
    assert sorted(r["pk"] for r in gamefeed.load(2026)) == [1, 2]
    assert gamefeed.fetch(2026, workers=1)["downloaded"] == 0  # nothing is downloaded twice
    assert gamefeed.fetch_pregame([3]) == 1 and gamefeed.load_pregame()[0]["pk"] == 3
    assert gamefeed.fetch_pregame([]) == 0 and gamefeed.load_pregame() == []


def _move(date, player, text, to=147, frm=None):
    item = {"date": date, "person": {"id": player}, "description": text, "toTeam": {"id": to}}
    if frm:
        item["fromTeam"] = {"id": frm}
    return item


def test_injured_list_moves_become_stretches_of_days(tmp_path, monkeypatch):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path))
    payload = {"transactions": [
        _move("2019-04-02", 1, "New York Yankees placed RHP A on the 10-day injured list. Sore elbow."),
        _move("2019-04-20", 1, "New York Yankees transferred RHP A from the 10-day injured list to the 60-day injured list."),
        _move("2019-06-05", 1, "New York Yankees activated RHP A from the 60-day injured list."),
        _move("2011-05-01", 2, "New York Yankees placed CF B on the 15-day disabled list retroactive to April 28, 2011."),
        _move("2011-05-10", 2, "New York Yankees released CF B.", frm=147),
        _move("2019-08-01", 3, "New York Yankees placed C C on the 10-day injured list."),
        _move("2019-08-02", 4, "New York Yankees optioned LHP D to Scranton."),
        _move("2019-08-03", None, "A move with no player."),
    ]}
    moves = transactions.slim(payload)
    assert [(m["player_id"], m["kind"], m["days"]) for m in moves] == [
        (1, "on", 10), (1, "on", 60), (1, "off", 60), (2, "on", 15), (2, "gone", 0), (3, "on", 10)]
    transactions._file(2019).write_text(json.dumps([m for m in moves if m["date"].startswith("2019")]))
    transactions._file(2011).write_text(json.dumps([m for m in moves if m["date"].startswith("2011")]))
    spells = transactions.spells([2011, 2019, 2020]).set_index("player_id")
    # A transfer to the longer list does not restart the stretch.
    assert spells.loc["1", "start"] == pd.Timestamp("2019-04-02") and spells.loc["1", "end"] == pd.Timestamp("2019-06-04")
    assert spells.loc["2", "end"] == pd.Timestamp("2011-05-09")  # released while on the list
    assert spells.loc["3", "end"] == pd.Timestamp("2019-11-05")  # never activated: out for the season
    assert set(spells["team"]) == {"NYY"}
    assert transactions.spells([2005]).empty


def test_statcast_tables_are_read_and_error_pages_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path))
    header = '"last_name, first_name","player_id","year","pa","bip","ba","est_ba","woba","est_woba"\n'
    savant._file("expected_batter", 2019).write_bytes(
        ("﻿" + header + '"Judge, Aaron","592450","2019","447","250",0.272,0.280,0.382,0.401\n').encode("utf-8"))
    savant._file("expected_pitcher", 2019).write_text(header + '"Cole, Gerrit","543037","2019","817","400",0.185,0.180,0.246,0.238\n')
    savant._file("outs_above_average", 2019).write_text(
        '"last_name, first_name","player_id","outs_above_average"\n"X, Y","1",7\n"X, Y","1",-2\n')
    savant._file("sprint_speed", 2019).write_text("<!DOCTYPE html><html>not a table</html>" + " " * 40)
    tables = savant.load([2018, 2019])
    expected = tables["expected"].set_index(["player_id", "role"])
    assert expected.loc[("592450", "bat"), "xwoba"] == pytest.approx(0.401)
    assert expected.loc[("543037", "pit"), "woba"] == pytest.approx(0.246)
    assert tables["fielding"]["oaa"].tolist() == [5]  # two positions for one player are added up
    assert tables["speed"].empty and list(tables["speed"].columns) == ["player_id", "season", "sprint_speed"]


def test_old_seasons_are_matched_to_mlb_game_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path))
    games = pd.DataFrame({
        "game_id": ["20190501-NYY-0", "20190502-NYY-1", "20190502-NYY-2", "20190503-LAD-0", "20190510-WSH-0",
                    "20260401-SF-0"],
        "date": pd.to_datetime(["2019-05-01", "2019-05-02", "2019-05-02", "2019-05-03", "2019-05-10", "2026-04-01"]),
        "season": [2019, 2019, 2019, 2019, 2019, 2026], "home": ["NYY", "NYY", "NYY", "LAD", "WSH", "SF"],
        "away": ["BOS", "BOS", "BOS", "SF", "TOR", "LAD"], "number": [0, 1, 2, 0, 0, 0],
        "status": "final", "mlb_game_pk": [None, None, None, None, None, 900.0],
    })

    def listed(pk, date, home, away, dh="N", number=1):
        return {"game_pk": pk, "official_date": date, "start_utc": None, "state": "Final", "double_header": dh,
                "game_number": number, "day_night": "night", "venue_id": 1, "home_id": home, "away_id": away,
                "home_score": 1, "away_score": 0, "home_sp_id": None, "home_sp_name": None,
                "away_sp_id": None, "away_sp_name": None}

    schedule = [listed(11, "2019-05-01", 147, 111), listed(12, "2019-05-02", 147, 111, "Y", 1),
                listed(13, "2019-05-02", 147, 111, "Y", 2),
                listed(14, "2019-05-04", 119, 137),  # MLB lists this one a day later than Retrosheet
                listed(15, "2019-05-10", 141, 120)]  # played in Washington with Toronto batting last
    (mlbapi._dir() / "schedule_2019.json").write_text(json.dumps(schedule))
    numbers = playerdata.game_numbers(games).set_index("game_id")["pk"]
    assert numbers.to_dict() == {"20190501-NYY-0": 11, "20190502-NYY-1": 12, "20190502-NYY-2": 13,
                                 "20190503-LAD-0": 14, "20190510-WSH-0": 15, "20260401-SF-0": 900}


def test_player_table_checks_catch_missing_lineups_and_wrong_starters():
    tables = make_league(seed=8, seasons=(2015,), days=20, scheduled=2)
    report, problems = playerdata.check(tables["games"], tables)
    assert problems == [] and report["starter_agreement"] == 1.0 and report["lineup_coverage"] == {2015: 1.0}
    broken = dict(tables)
    broken["lineups"] = tables["lineups"].iloc[: len(tables["lineups"]) // 2]
    assert "nine-man starting lineup" in playerdata.check(tables["games"], broken)[1][0]
    broken = dict(tables)
    wrong = tables["plate_appearances"].copy()
    wrong.loc[wrong["vs_starter"], "pitcher"] = "someone else"
    broken["plate_appearances"] = wrong
    assert "same starting pitcher" in " ".join(playerdata.check(tables["games"], broken)[1])


def test_tables_are_keyed_like_the_rest_of_the_project():
    raw = gamefeed.tables_from([gamefeed.slim(_payload())])
    games = pd.DataFrame({"game_id": ["20260615-NYY-0"], "status": ["final"]})
    numbers = pd.DataFrame({"game_id": ["20260615-NYY-0"], "pk": [777]})
    tables = playerdata.assemble(games, [raw], numbers)
    assert set(tables["lineups"]["team"]) == {"NYY", "BOS"}
    assert set(tables["lineups"].loc[tables["lineups"]["team"] == "NYY", "player_id"]) == {str(p) for p in range(101, 110)}
    assert set(tables["plate_appearances"]["bat_team"]) == {"NYY", "BOS"}
    extras = tables["game_extras"].iloc[0]
    assert extras["wind_out"] == 9.0 and extras["temp"] == 81 and extras["roof_closed"] == 0.0
    assert tables["people"]["player_id"].tolist() == ["101"]


def test_a_season_table_saved_mid_season_is_fetched_again(tmp_path, monkeypatch):
    import os
    import datetime as dt
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path))
    calls = []

    def fake_download(url, dest, *, refresh=False, allow_404=False):
        calls.append((dest.name, refresh))
        dest.write_text("player_id\n1\n")
        return True

    monkeypatch.setattr(savant, "download", fake_download)
    assert savant.fetch(2019) == 4 and all(not refresh for _, refresh in calls)
    calls.clear()
    assert savant.fetch(2019) == 4 and all(not refresh for _, refresh in calls)  # saved long after 2019 ended
    july = dt.datetime(2019, 7, 1).timestamp()
    os.utime(savant._file("sprint_speed", 2019), (july, july))
    calls.clear()
    savant.fetch(2019)
    assert dict(calls)["sprint_speed_2019.csv"] is True and dict(calls)["expected_batter_2019.csv"] is False

    served = []
    monkeypatch.setattr(transactions, "get_json", lambda url: served.append(url) or {"transactions": []})
    assert transactions.fetch(2019) == [] and len(served) == 10  # February to November
    assert transactions.fetch(2019) == [] and len(served) == 10  # finished season: not asked again
    os.utime(transactions._file(2019), (july, july))
    transactions.fetch(2019)
    assert len(served) == 20
