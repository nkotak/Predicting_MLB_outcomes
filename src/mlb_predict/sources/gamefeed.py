"""Player-level detail for every game, from MLB's public game feed.

One request per game returns:

* the batting order of both teams, with each player's fielding position
* every plate appearance: who batted, who pitched, which side each one bats or
  throws from, what happened, and a short summary of the pitches
* who was on the active roster that day (bench and bullpen included)
* the weather and the home plate umpire

The feed goes back further than this project needs and uses MLB's own player
ids, so the same code reads a game from 2010 and a game from tonight. Only a
slim version of each game is kept on disk.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import ensure, raw_dir
from ..http import FetchError, get_json
from . import mlbapi

log = logging.getLogger(__name__)

# The feed only sends the fields named here, which keeps each download small.
FIELDS = ",".join([
    "gamePk", "gameData", "status", "detailedState", "datetime", "officialDate",
    "weather", "condition", "temp", "wind", "venue", "id", "teams", "home", "away",
    "players", "batSide", "pitchHand", "code", "birthDate", "fullName", "primaryPosition",
    "liveData", "boxscore", "battingOrder", "person", "batters", "pitchers", "bench", "bullpen",
    "position", "allPositions", "abbreviation", "officials", "official", "officialType",
    "plays", "allPlays", "result", "eventType", "about", "inning", "halfInning", "matchup",
    "batter", "pitcher", "playEvents", "isPitch", "details", "call", "type",
    "pitchData", "startSpeed", "hitData", "launchSpeed", "launchAngle", "trajectory",
])
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?fields=" + FIELDS

# What a plate appearance ended in. 0 means "not a plate appearance" (a runner
# thrown out to end an inning, for example); its pitches still count.
EVENTS = {
    "single": 1, "double": 2, "triple": 3, "home_run": 4,
    "walk": 5, "intent_walk": 6, "hit_by_pitch": 7,
    "strikeout": 8, "strikeout_double_play": 8, "strikeout_triple_play": 8,
    "sac_fly": 9, "sac_fly_double_play": 9,
    "sac_bunt": 10, "sac_bunt_double_play": 10,
    "catcher_interf": 11,
    "field_error": 12,
    "field_out": 13, "force_out": 13, "grounded_into_double_play": 13, "double_play": 13,
    "triple_play": 13, "fielders_choice": 13, "fielders_choice_out": 13,
}
WHIFF = {"S", "W", "T", "M", "Q"}       # swung and missed (foul tips included)
CALLED = {"C"}
FOUL = {"F", "L", "O", "R"}
IN_PLAY = {"X", "D", "E"}
FASTBALLS = {"FF", "FT", "SI", "FA"}
TRAJECTORY = {"ground_ball": "G", "line_drive": "L", "fly_ball": "F", "popup": "P",
              "bunt_grounder": "B", "bunt_popup": "B", "bunt_line_drive": "B"}

#: Column order of one stored plate appearance.
PA_FIELDS = ("half", "inning", "batter", "pitcher", "bat_side", "pit_hand", "event",
             "pitches", "whiffs", "called", "fouls", "balls", "in_play",
             "fb_speed_sum", "fb_n", "launch_speed", "launch_angle", "trajectory")

_write_lock = threading.Lock()


def _dir() -> Path:
    return ensure(raw_dir() / "mlb" / "feed")


def _file(season: int) -> Path:
    return _dir() / f"feed_{season}.jsonl"


def _int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _slim_play(play: dict) -> list | None:
    matchup = play.get("matchup") or {}
    batter = (matchup.get("batter") or {}).get("id")
    pitcher = (matchup.get("pitcher") or {}).get("id")
    if batter is None or pitcher is None:
        return None
    pitches = whiffs = called = fouls = balls = in_play = fb_n = 0
    fb_sum = 0.0
    speed = angle = None
    trajectory = ""
    for event in play.get("playEvents") or []:
        if not event.get("isPitch"):
            continue
        details = event.get("details") or {}
        code = (details.get("call") or {}).get("code") or ""
        pitches += 1
        if code in WHIFF:
            whiffs += 1
        elif code in CALLED:
            called += 1
        elif code in FOUL:
            fouls += 1
        elif code in IN_PLAY:
            in_play += 1
        else:
            balls += 1
        velocity = (event.get("pitchData") or {}).get("startSpeed")
        if velocity is not None and (details.get("type") or {}).get("code") in FASTBALLS:
            fb_sum += float(velocity)
            fb_n += 1
        hit = event.get("hitData")
        if hit:
            speed = hit.get("launchSpeed", speed)
            angle = hit.get("launchAngle", angle)
            trajectory = TRAJECTORY.get(hit.get("trajectory") or "", trajectory)
    about = play.get("about") or {}
    return [
        0 if about.get("halfInning") == "top" else 1, _int(about.get("inning")), batter, pitcher,
        (matchup.get("batSide") or {}).get("code") or "", (matchup.get("pitchHand") or {}).get("code") or "",
        EVENTS.get((play.get("result") or {}).get("eventType") or "", 0),
        pitches, whiffs, called, fouls, balls, in_play, round(fb_sum, 1), fb_n, speed, angle, trajectory,
    ]



def _starting_lineup(block: dict) -> list[list]:
    """The nine starters in batting order, each with the position he started at.

    After a game the feed's own ``battingOrder`` list holds whoever finished
    the game in each slot, so starters are read from each player's batting
    order code instead: "300" is the starter in the third slot, "301" his
    first replacement. Before a game only the posted list exists, and it is
    the starting lineup.
    """
    players = block.get("players") or {}

    def position(entry: dict) -> str:
        first = (entry.get("allPositions") or [{}])[0]
        return first.get("abbreviation") or (entry.get("position") or {}).get("abbreviation") or ""

    starters = {}
    for entry in players.values():
        code = str(entry.get("battingOrder") or "")
        if len(code) == 3 and code.endswith("00") and code[0] in "123456789":
            starters[int(code[0])] = [int((entry.get("person") or {}).get("id")), position(entry)]
    if len(starters) == 9:
        return [starters[slot] for slot in range(1, 10)]
    posted = [int(p) for p in block.get("battingOrder") or []]
    if len(posted) == 9:
        return [[p, position(players.get(f"ID{p}") or {})] for p in posted]
    return []


def slim(payload: dict) -> dict:
    """Reduce one game's feed to the parts this project uses."""
    game = payload.get("gameData") or {}
    live = payload.get("liveData") or {}
    box = live.get("boxscore") or {}
    weather = game.get("weather") or {}
    plate_umpire = next((o.get("official", {}).get("id") for o in box.get("officials") or []
                         if o.get("officialType") == "Home Plate"), None)
    sides = {}
    for side in ("home", "away"):
        block = (box.get("teams") or {}).get(side) or {}
        sides[side] = {
            "team": ((game.get("teams") or {}).get(side) or {}).get("id"),
            "lineup": _starting_lineup(block),
            "batters": [int(p) for p in block.get("batters") or []],
            "pitchers": [int(p) for p in block.get("pitchers") or []],
            "bullpen": [int(p) for p in block.get("bullpen") or []],
            "bench": [int(p) for p in block.get("bench") or []],
        }
    people = {}
    for person in (game.get("players") or {}).values():
        if person.get("id") is None:
            continue
        people[str(person["id"])] = [
            person.get("fullName") or "",
            (person.get("batSide") or {}).get("code") or "",
            (person.get("pitchHand") or {}).get("code") or "",
            person.get("birthDate") or "",
            (person.get("primaryPosition") or {}).get("abbreviation") or "",
        ]
    plays = [row for row in map(_slim_play, (live.get("plays") or {}).get("allPlays") or []) if row]
    return {
        "pk": payload.get("gamePk"),
        "date": (game.get("datetime") or {}).get("officialDate"),
        "state": (game.get("status") or {}).get("detailedState"),
        "venue": (game.get("venue") or {}).get("id"),
        "temp": _int(weather.get("temp")),
        "wind": weather.get("wind") or "",
        "sky": weather.get("condition") or "",
        "umpire": plate_umpire,
        "sides": sides,
        "people": people,
        "plays": plays,
    }


def fetch_game(game_pk: int) -> dict:
    """One game's slim feed, downloaded now and not stored. Used for upcoming games."""
    record = slim(get_json(FEED_URL.format(pk=game_pk)))
    record["pk"] = int(game_pk)
    return record


def stored_pks(season: int) -> set[int]:
    path = _file(season)
    found: set[int] = set()
    if path.exists():
        with path.open() as handle:
            for line in handle:
                # The game number is the first field of each line, so the whole
                # line does not have to be parsed to know which games are stored.
                head = line[:24]
                if head.startswith('{"pk": '):
                    found.add(int(head[7:].split(",")[0]))
    return found


def load(season: int) -> list[dict]:
    """Every stored game of one season. The last copy of a game wins."""
    path = _file(season)
    games: dict[int, dict] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    games[int(record["pk"])] = record
    return list(games.values())


def _fetch_and_store(game_pk: int, season: int) -> bool:
    try:
        record = fetch_game(game_pk)
    except FetchError as exc:
        log.warning("game feed %s failed: %s", game_pk, exc)
        return False
    with _write_lock, _file(season).open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return True


def fetch(season: int, *, workers: int = 8) -> dict:
    """Download the feed for finished games of ``season`` that are not stored yet.

    Safe to stop and run again. A game that fails to download is skipped and
    picked up by the next run.
    """
    schedule = mlbapi.fetch_schedule(season)
    have = stored_pks(season)
    todo = sorted({g["game_pk"] for g in schedule if g["state"] in mlbapi.FINAL_STATES} - have)
    log.info("game feed %d: %d stored, %d to download", season, len(have), len(todo))
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = sum(pool.map(lambda pk: _fetch_and_store(pk, season), todo))
    return {"season": season, "stored_before": len(have), "downloaded": done, "failed": len(todo) - done}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

# What a walk, a single and so on are worth, on the scale of "weighted on-base
# average" (wOBA). One number per hitter that counts a home run for more than
# a single. The same fixed weights are used for every season.
WOBA_WEIGHTS = {"bb": 0.69, "hbp": 0.72, "h1": 0.88, "h2": 1.25, "h3": 1.58, "hr": 2.03}

#: Counted columns of the plate appearance table.
PA_SUMS = ("pa", "ab", "h1", "h2", "h3", "hr", "bb", "ibb", "hbp", "so", "sf", "woba_num", "woba_den",
           "pitches", "whiffs", "called", "swings", "fb_speed_sum", "fb_n",
           "bip", "bip_hit", "tracked", "gb", "ev_sum", "ev_n", "hard")


def _pregame_file() -> Path:
    return _dir() / "pregame.jsonl"


def fetch_pregame(game_pks: list[int], *, workers: int = 6) -> int:
    """Download the feed for upcoming games and replace the stored pre-game file.

    Before a game the feed carries the active roster and, once it is posted a
    few hours ahead, the starting lineup.
    """
    def one(pk: int) -> dict | None:
        try:
            return fetch_game(pk)
        except FetchError as exc:
            log.warning("pre-game feed %s failed: %s", pk, exc)
            return None

    records = []
    if game_pks:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = [r for r in pool.map(one, sorted(set(game_pks))) if r is not None]
    _pregame_file().write_text("".join(json.dumps(r) + "\n" for r in records))
    return len(records)


def load_pregame() -> list[dict]:
    path = _pregame_file()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse(season: int):
    """One stored season as tables. See ``tables_from``."""
    return tables_from(load(season))


def tables_from(records: list[dict]):
    """Slim game records as tables keyed by MLB's game number (``pk``).

    ``lineups``             the nine starters of each team, in batting order
    ``rosters``             everyone on the active roster that day, with a role
    ``plate_appearances``   totals for each batter against each pitcher in a game
    ``game_extras``         weather, home plate umpire and ballpark
    ``people``              name, batting side, throwing hand, birth date
    """
    import numpy as np
    import pandas as pd

    lineup_rows, roster_rows, extra_rows, play_rows, people = [], [], [], [], {}
    for record in records:
        pk = int(record["pk"])
        extra_rows.append((pk, record.get("date"), record.get("venue"), record.get("temp"),
                           record.get("wind") or "", record.get("sky") or "", record.get("umpire")))
        teams_by_half = {}
        for side, half in (("away", 0), ("home", 1)):
            block = record["sides"][side]
            team = block["team"]
            teams_by_half[half] = team
            for slot, (player, position) in enumerate(block["lineup"], start=1):
                lineup_rows.append((pk, team, slot, player, position))
            pitchers = block["pitchers"]
            staff = set(pitchers) | set(block["bullpen"])
            for seq, player in enumerate(pitchers):
                roster_rows.append((pk, team, player, "SP" if seq == 0 else "RP"))
            roster_rows.extend((pk, team, player, "BP") for player in block["bullpen"] if player not in pitchers)
            seen = set(staff)
            for role, group in (("BAT", block["batters"]), ("BN", block["bench"])):
                for player in group:
                    if player not in seen:
                        seen.add(player)
                        roster_rows.append((pk, team, player, role))
        people.update(record.get("people") or {})
        first_pitcher = {record["sides"]["home"]["team"]: (record["sides"]["home"]["pitchers"] or [None])[0],
                         record["sides"]["away"]["team"]: (record["sides"]["away"]["pitchers"] or [None])[0]}
        for play in record["plays"]:
            bat_team, pit_team = teams_by_half[play[0]], teams_by_half[1 - play[0]]
            play_rows.append((pk, bat_team, pit_team, play[3] == first_pitcher.get(pit_team), *play[2:]))

    lineups = pd.DataFrame(lineup_rows, columns=["pk", "team_id", "slot", "player_id", "position"])
    rosters = pd.DataFrame(roster_rows, columns=["pk", "team_id", "player_id", "role"])
    extras = pd.DataFrame(extra_rows, columns=["pk", "date", "venue", "temp", "wind", "sky", "umpire"])
    person = pd.DataFrame(
        [(int(pid), *values) for pid, values in people.items()],
        columns=["player_id", "name", "bats", "throws", "birth_date", "position"])

    plays = pd.DataFrame(play_rows, columns=["pk", "bat_team_id", "pit_team_id", "vs_starter", *PA_FIELDS[2:]])
    event = plays["event"].to_numpy() if len(plays) else np.array([], dtype=int)
    out = plays[["pk", "bat_team_id", "pit_team_id", "vs_starter", "batter", "pitcher", "bat_side", "pit_hand"]].copy()
    for name, code in (("h1", 1), ("h2", 2), ("h3", 3), ("hr", 4), ("bb", 5), ("ibb", 6), ("hbp", 7),
                       ("so", 8), ("sf", 9)):
        out[name] = (event == code).astype(float)
    out["pa"] = (event > 0).astype(float)
    not_at_bat = np.isin(event, (5, 6, 7, 9, 10, 11))
    out["ab"] = ((event > 0) & ~not_at_bat).astype(float)
    out["woba_num"] = sum(weight * out[name] for name, weight in WOBA_WEIGHTS.items())
    out["woba_den"] = out["ab"] + out["bb"] + out["sf"] + out["hbp"]
    for name in ("pitches", "whiffs", "called", "fb_speed_sum", "fb_n"):
        out[name] = plays[name].astype(float)
    out["swings"] = (plays["whiffs"] + plays["fouls"] + plays["in_play"]).astype(float)
    # Balls put in play that a fielder could turn into an out (home runs excluded).
    out["bip"] = np.isin(event, (1, 2, 3, 9, 10, 12, 13)).astype(float)
    out["bip_hit"] = np.isin(event, (1, 2, 3)).astype(float)
    trajectory = plays["trajectory"].fillna("")
    out["tracked"] = trajectory.isin(["G", "L", "F", "P"]).astype(float)
    out["gb"] = (trajectory == "G").astype(float)
    speed = pd.to_numeric(plays["launch_speed"], errors="coerce")
    out["ev_sum"] = speed.fillna(0.0)
    out["ev_n"] = speed.notna().astype(float)
    out["hard"] = (speed >= 95.0).astype(float)
    keys = ["pk", "bat_team_id", "pit_team_id", "batter", "pitcher", "bat_side", "pit_hand"]
    out["vs_starter"] = out["vs_starter"].astype(float)
    grouped = out.groupby(keys, as_index=False, sort=False).agg(
        {"vs_starter": "max", **{name: "sum" for name in PA_SUMS}})
    grouped["vs_starter"] = grouped["vs_starter"] > 0
    return {"lineups": lineups, "rosters": rosters, "plate_appearances": grouped,
            "game_extras": extras, "people": person}
