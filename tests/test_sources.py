"""Reading the three data sources, using small hand-made samples. No network."""

import io
import zipfile

import pandas as pd
import pytest

from mlb_predict import teams
from mlb_predict.sources import espn, mlbapi, retrosheet


# ---------------------------------------------------------------- teams


def test_team_table_is_complete_and_consistent():
    assert len(teams.TEAMS) == 30
    assert len(set(teams.CODES)) == 30
    assert len({t.mlb_id for t in teams.TEAMS}) == 30
    assert len({t.espn_id for t in teams.TEAMS}) == 30
    # The clubs that tripped up the earlier version of this project.
    assert teams.from_retro("WAS") == "WSH" and teams.from_mlb_id(120) == "WSH" and teams.from_espn_id(20) == "WSH"
    assert teams.from_retro("ARI") == "ARI" and teams.from_mlb_id(109) == "ARI"
    assert teams.from_retro("OAK") == teams.from_retro("ATH") == teams.from_mlb_id(133) == "ATH"
    assert teams.from_retro("FLO") == teams.from_retro("MIA") == "MIA"


def test_unknown_team_codes_raise_instead_of_passing_through():
    with pytest.raises(teams.UnknownTeam):
        teams.from_retro("XYZ")
    with pytest.raises(teams.UnknownTeam):
        teams.from_mlb_id(999)
    with pytest.raises(teams.UnknownTeam):
        teams.from_espn_id(0)


# ---------------------------------------------------------------- ESPN odds


def _book(name, home, away, close=None, opening=None):
    def side(price, which):
        block = {"moneyLine": price}
        if close:
            block["close"] = {"moneyLine": {"american": close[which]}}
        if opening:
            block["open"] = {"moneyLine": {"american": opening[which]}}
        return block
    return {"provider": {"name": name}, "homeTeamOdds": side(home, 0), "awayTeamOdds": side(away, 1)}


def test_american_prices_are_parsed():
    assert espn._american(-160) == -160.0
    assert espn._american("+135") == 135.0
    assert espn._american("EVEN") == 100.0
    assert espn._american(0) is None  # ESPN stores 0 when a book has no price
    assert espn._american(None) is None and espn._american("off") is None


def test_in_game_prices_are_never_used():
    payload = {"items": [
        _book("ESPN BET", 132, -160, close=("+132", "-160")),
        _book("ESPN BET - Live Odds", -900, 600),  # taken late in a game the home team is winning
    ]}
    price = espn.consensus(espn.slim_odds(payload))
    assert price["n_books"] == 1
    assert price["home_dec"] == pytest.approx(2.32)
    assert 0.38 < price["market_home_prob"] < 0.42


def test_closing_price_is_preferred_and_broken_markets_are_ignored():
    payload = {"items": [
        _book("DraftKings", -105, -105, close=("+120", "-140")),  # stored price drifted; close is the truth
        _book("Bet365", 0, 0),                                    # no price
        _book("BOVADA.lv", 300, 300),                             # not a real two-way market
        _book("consensus", 110, -130),                            # an average, not a sportsbook
    ]}
    books = espn.slim_odds(payload)
    assert books[0]["home_close"] == 120.0 and books[1]["home"] is None
    price = espn.consensus(books)
    assert price["n_books"] == 1 and price["home_dec"] == pytest.approx(2.2)
    assert espn.consensus(espn.slim_odds({"items": [_book("Bet365", 0, 0)]})) is None


def test_consensus_takes_the_middle_book():
    payload = {"items": [_book(f"Book {i}", h, a) for i, (h, a) in enumerate([(-150, 130), (-140, 120), (-400, 300)])]}
    price = espn.consensus(espn.slim_odds(payload))
    assert price["n_books"] == 3
    assert price["home_dec"] == pytest.approx(1 + 100 / 150)  # the median, not pulled by the outlier


def test_scoreboard_is_reduced_to_what_is_needed():
    payload = {"events": [{
        "id": "401", "date": "2026-09-20T17:10Z", "season": {"type": 2},
        "competitions": [{"status": {"type": {"name": "STATUS_FINAL"}}, "competitors": [
            {"homeAway": "home", "team": {"id": "21"}, "score": "3"},
            {"homeAway": "away", "team": {"id": "22"}, "score": "5"}]}],
    }]}
    events = espn.slim_scoreboard(payload, "20260920")
    assert events == [{"event_id": "401", "query_date": "20260920", "start_utc": "2026-09-20T17:10Z",
                       "season_type": 2, "status": "STATUS_FINAL", "home_espn_id": 21, "away_espn_id": 22,
                       "home_score": "3", "away_score": "5"}]


# ---------------------------------------------------------------- MLB schedule


def _game(pk, date, state, home, away, **extra):
    return {"gamePk": pk, "officialDate": date, "gameDate": f"{date}T23:05:00Z",
            "status": {"detailedState": state}, "doubleHeader": extra.get("dh", "N"),
            "gameNumber": extra.get("number", 1), "dayNight": "night", "venue": {"id": 1},
            "teams": {"home": {"team": {"id": home}, "score": extra.get("hs"),
                               "probablePitcher": extra.get("hp")},
                      "away": {"team": {"id": away}, "score": extra.get("as_")}}}


def test_schedule_handles_finals_postponements_doubleheaders_and_upcoming_games():
    payload = {"dates": [
        {"date": "2026-05-01", "games": [
            _game(1, "2026-05-01", "Final", 147, 111, hs=5, as_=3, hp={"id": 543037, "fullName": "Gerrit Cole"}),
            _game(2, "2026-05-01", "Postponed", 121, 143),
        ]},
        {"date": "2026-05-02", "games": [
            _game(2, "2026-05-02", "Final", 121, 143, hs=1, as_=2, dh="Y", number=1),
            _game(3, "2026-05-02", "Final", 121, 143, hs=4, as_=0, dh="Y", number=2),
            _game(4, "2026-05-02", "Scheduled", 119, 137),
        ]},
    ]}
    frame = mlbapi.schedule_frame(mlbapi.slim_schedule(payload))
    assert list(frame["game_id"]) == ["20260501-NYY-0", "20260502-LAD-0", "20260502-NYM-1", "20260502-NYM-2"]
    assert list(frame["status"]) == ["final", "scheduled", "final", "final"]
    assert frame.loc[0, "home_score"] == 5 and frame.loc[0, "home_sp_id"] == 543037
    assert pd.isna(frame.loc[1, "home_score"])  # an unplayed game has no score
    assert frame["mlb_game_pk"].is_unique  # the postponed listing of game 2 was dropped


def test_box_score_is_reduced_to_team_totals_and_pitcher_lines():
    def side(team_id, pitchers):
        players = {f"ID{pid}": {"person": {"fullName": f"P{pid}"},
                                "stats": {"pitching": {"outs": outs, "battersFaced": outs + 4, "hits": 3,
                                                       "homeRuns": 0, "runs": 1, "earnedRuns": 1, "baseOnBalls": 1,
                                                       "intentionalWalks": 0, "strikeOuts": 5, "hitBatsmen": 0,
                                                       "gamesStarted": gs}}}
                   for pid, outs, gs in pitchers}
        return {"team": {"id": team_id}, "pitchers": [p[0] for p in pitchers], "players": players,
                "teamStats": {"batting": {"plateAppearances": 38, "atBats": 33, "runs": 4, "hits": 9},
                              "pitching": {"outs": 27, "battersFaced": 36}, "fielding": {"errors": 1}}}
    box = mlbapi.slim_boxscore({"teams": {"home": side(147, [(10, 18, 1), (11, 9, 0)]),
                                          "away": side(111, [(20, 15, 1), (21, 9, 0)])}}, 777)
    home = box["teams"]["home"]
    assert home["stats"]["runs"] == 4 and home["stats"]["p_outs"] == 27 and home["stats"]["errors"] == 1
    assert [p["is_start"] for p in home["pitchers"]] == [True, False]
    assert home["pitchers"][0]["outs"] == 18 and home["pitchers"][1]["seq"] == 2


# ---------------------------------------------------------------- Retrosheet


def test_retrosheet_season_is_read_into_standard_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("MLB_DATA_DIR", str(tmp_path))
    info = ("gid,visteam,hometeam,site,date,number,daynight,gametype,vruns,hruns\n"
            "WAS202204070,NYN,WAS,WAS11,20220407,0,night,regular,5,1\n"
            "WAS202210010,PHI,WAS,WAS11,20221001,1,day,regular,2,3\n"
            "HOU202210280,PHI,HOU,HOU03,20221028,0,night,worldseries,6,5\n")
    team_cols = list(retrosheet.TEAM_COLUMNS)
    stats = "gid,team,stattype,gametype,vishome," + ",".join(team_cols) + "\n"
    for gid, team, where, runs in (("WAS202204070", "NYN", "v", 5), ("WAS202204070", "WAS", "h", 1),
                                   ("WAS202210010", "PHI", "v", 2), ("WAS202210010", "WAS", "h", 3)):
        values = ["4"] * len(team_cols)
        values[team_cols.index("b_r")] = str(runs)
        stats += f"{gid},{team},value,regular,{where}," + ",".join(values) + "\n"
    stats += "WAS202204070,WAS,official,regular,h," + ",".join(["9"] * len(team_cols)) + "\n"
    pitch_cols = list(retrosheet.PITCHER_COLUMNS)
    pitching = "gid,id,team,p_seq,stattype,gametype,p_gs," + ",".join(pitch_cols) + "\n"
    pitching += "WAS202204070,schem001,NYN,1,value,regular,1," + ",".join(["6"] * len(pitch_cols)) + "\n"
    pitching += "WAS202204070,corbp001,WAS,1,value,regular,1," + ",".join(["5"] * len(pitch_cols)) + "\n"
    pitching += "WAS202204070,doe--001,WAS,2,value,regular,," + ",".join(["1"] * len(pitch_cols)) + "\n"
    people = "id,last,first,throw\nschem001,Scherzer,Max,R\ncorbp001,Corbin,Patrick,L\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in (("gameinfo", info), ("teamstats", stats), ("pitching", pitching), ("allplayers", people)):
            archive.writestr(f"2022{name}.csv", text)
    retrosheet.zip_path(2022).write_bytes(buffer.getvalue())

    tables = retrosheet.parse(2022)
    games = tables["games"]
    assert list(games["game_id"]) == ["20220407-WSH-0", "20221001-WSH-1"]  # the World Series game is left out
    assert list(games["away"]) == ["NYM", "PHI"] and games.loc[0, "day_night"] == "N"
    team_games = tables["team_games"]
    assert len(team_games) == 4  # the "official" duplicate row is ignored
    assert team_games.loc[(team_games["game_id"] == "20220407-WSH-0") & team_games["is_home"], "runs"].item() == 1
    pitchers = tables["pitcher_games"]
    assert pitchers["is_start"].tolist() == [True, True, False]
    assert pitchers.loc[0, "outs"] == 6 and pitchers.loc[0, "player_id"] == "schem001"
    assert tables["players"].set_index("player_id").loc["schem001", "name"] == "Max Scherzer"
