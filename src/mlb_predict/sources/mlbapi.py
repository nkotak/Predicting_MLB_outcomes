"""Current-season games and upcoming schedules from MLB's public feed.

Retrosheet covers finished seasons. This module fills the gap for the season
in progress (or one Retrosheet has not published yet) and supplies the list of
upcoming games with their announced starting pitchers.

Players are identified by MLB's own numeric ids.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .. import teams
from ..config import ensure, raw_dir
from ..http import get_json

log = logging.getLogger(__name__)

SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R"
    "&startDate={start}&endDate={end}&hydrate=probablePitcher"
)
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"

FINAL_STATES = {"Final", "Completed Early", "Game Over"}
DEAD_STATES = {"Postponed", "Cancelled", "Suspended"}

_write_lock = threading.Lock()

BATTING = {
    "plateAppearances": "pa", "atBats": "ab", "runs": "runs", "hits": "h", "doubles": "doubles",
    "triples": "triples", "homeRuns": "hr", "baseOnBalls": "bb", "intentionalWalks": "ibb",
    "hitByPitch": "hbp", "strikeOuts": "so", "sacFlies": "sf", "stolenBases": "sb",
    "caughtStealing": "cs", "leftOnBase": "lob",
}
TEAM_PITCHING = {
    "outs": "p_outs", "battersFaced": "p_bf", "hits": "p_h", "homeRuns": "p_hr", "runs": "p_r",
    "earnedRuns": "p_er", "baseOnBalls": "p_bb", "strikeOuts": "p_so", "hitBatsmen": "p_hbp",
}
PITCHER = {
    "outs": "outs", "battersFaced": "bf", "hits": "h", "homeRuns": "hr", "runs": "r",
    "earnedRuns": "er", "baseOnBalls": "bb", "intentionalWalks": "ibb", "strikeOuts": "so",
    "hitBatsmen": "hbp",
}


def _dir() -> Path:
    return ensure(raw_dir() / "mlb")


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------


def slim_schedule(payload: dict) -> list[dict]:
    games = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            home, away = game["teams"]["home"], game["teams"]["away"]
            games.append({
                "game_pk": game["gamePk"],
                "official_date": game.get("officialDate") or day["date"],
                "start_utc": game.get("gameDate"),
                "state": game["status"].get("detailedState"),
                "double_header": game.get("doubleHeader"),
                "game_number": game.get("gameNumber"),
                "day_night": game.get("dayNight"),
                "venue_id": (game.get("venue") or {}).get("id"),
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_sp_id": (home.get("probablePitcher") or {}).get("id"),
                "home_sp_name": (home.get("probablePitcher") or {}).get("fullName"),
                "away_sp_id": (away.get("probablePitcher") or {}).get("id"),
                "away_sp_name": (away.get("probablePitcher") or {}).get("fullName"),
            })
    return games


def fetch_schedule(season: int, *, refresh: bool = False) -> list[dict]:
    """Every regular-season game on the schedule for ``season``."""
    path = _dir() / f"schedule_{season}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    payload = get_json(SCHEDULE_URL.format(start=f"{season}-02-15", end=f"{season}-11-30"))
    games = slim_schedule(payload)
    if not games:
        log.warning("MLB schedule for %d is empty", season)
    path.write_text(json.dumps(games))
    return games


def fetch_day(date: dt.date) -> list[dict]:
    """The schedule for one date, always downloaded fresh (starters change late)."""
    payload = get_json(SCHEDULE_URL.format(start=date.isoformat(), end=date.isoformat()))
    return slim_schedule(payload)


# --------------------------------------------------------------------------
# Box scores
# --------------------------------------------------------------------------


def slim_boxscore(payload: dict, game_pk: int) -> dict:
    out = {"game_pk": game_pk, "teams": {}}
    for side in ("home", "away"):
        block = payload["teams"][side]
        stats = block.get("teamStats") or {}
        batting, pitching = stats.get("batting") or {}, stats.get("pitching") or {}
        row = {target: batting.get(source) for source, target in BATTING.items()}
        row.update({target: pitching.get(source) for source, target in TEAM_PITCHING.items()})
        row["errors"] = (stats.get("fielding") or {}).get("errors")
        pitchers = []
        for seq, pid in enumerate(block.get("pitchers") or [], start=1):
            player = (block.get("players") or {}).get(f"ID{pid}") or {}
            line = (player.get("stats") or {}).get("pitching") or {}
            record = {target: line.get(source) for source, target in PITCHER.items()}
            record.update({
                "player_id": pid,
                "name": (player.get("person") or {}).get("fullName"),
                "seq": seq,
                "is_start": int(line.get("gamesStarted") or 0) == 1,
            })
            pitchers.append(record)
        out["teams"][side] = {"team_id": block["team"]["id"], "stats": row, "pitchers": pitchers}
    return out


def _box_file(season: int) -> Path:
    return _dir() / f"boxscores_{season}.jsonl"


def load_boxscores(season: int) -> dict[int, dict]:
    path = _box_file(season)
    boxes: dict[int, dict] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    boxes[int(record["game_pk"])] = record
    return boxes


def _fetch_box(game_pk: int, season: int) -> None:
    record = slim_boxscore(get_json(BOXSCORE_URL.format(pk=game_pk)), game_pk)
    with _write_lock, _box_file(season).open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def fetch_boxscores(season: int, *, workers: int = 6) -> int:
    """Download box scores for finished games that are not stored yet."""
    schedule = fetch_schedule(season)
    have = load_boxscores(season)
    todo = sorted({g["game_pk"] for g in schedule if g["state"] in FINAL_STATES} - set(have))
    log.info("MLB box scores %d: %d stored, %d to download", season, len(have), len(todo))
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda pk: _fetch_box(pk, season), todo))
    return len(todo)


# --------------------------------------------------------------------------
# Standard tables
# --------------------------------------------------------------------------


def _number(game: dict) -> int:
    """0 for a single game, 1 or 2 for the games of a doubleheader."""
    return int(game.get("game_number") or 1) if game.get("double_header") in ("Y", "S") else 0


def schedule_frame(schedule: list[dict]) -> pd.DataFrame:
    """One row per game from a slim schedule, finished and upcoming games alike.

    A postponed game stays in MLB's feed under its original date. Those rows
    are dropped. The make-up game appears under its new date.
    """
    rows = []
    for game in schedule:
        if game["state"] in DEAD_STATES:
            continue
        final = game["state"] in FINAL_STATES
        date = pd.Timestamp(game["official_date"])
        home, away = teams.from_mlb_id(game["home_id"]), teams.from_mlb_id(game["away_id"])
        number = _number(game)
        rows.append({
            "game_id": f"{date:%Y%m%d}-{home}-{number}",
            "date": date,
            "season": date.year,
            "number": number,
            "home": home,
            "away": away,
            "home_score": game["home_score"] if final else None,
            "away_score": game["away_score"] if final else None,
            "status": "final" if final else "scheduled",
            "day_night": {"day": "D", "night": "N"}.get(game.get("day_night")),
            "park": str(game.get("venue_id") or ""),
            "source": "mlbapi",
            "mlb_game_pk": game["game_pk"],
            "start_utc": game.get("start_utc"),
            "home_sp_id": game.get("home_sp_id"),
            "away_sp_id": game.get("away_sp_id"),
            "home_sp_name": game.get("home_sp_name"),
            "away_sp_name": game.get("away_sp_name"),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # A game pk can be listed twice (a suspended game finished on a later
    # date). Keep the finished listing.
    frame = frame.sort_values(["mlb_game_pk", "status"]).drop_duplicates("mlb_game_pk", keep="first")
    return frame.sort_values(["date", "number", "home"]).reset_index(drop=True)


def parse(season: int) -> dict[str, pd.DataFrame]:
    """One season from MLB's feed in the project's standard tables."""
    games = schedule_frame(fetch_schedule(season))
    boxes = load_boxscores(season)
    team_rows, pitcher_rows, player_rows = [], [], {}
    for game in games.itertuples(index=False):
        if game.status != "final":
            continue
        box = boxes.get(int(game.mlb_game_pk))
        if box is None:
            raise RuntimeError(
                f"box score for finished game {game.game_id} (pk {game.mlb_game_pk}) is missing. "
                "Run `mlb fetch` again."
            )
        for side, is_home in (("home", True), ("away", False)):
            block = box["teams"][side]
            team = teams.from_mlb_id(block["team_id"])
            team_rows.append({"game_id": game.game_id, "team": team, "is_home": is_home, **block["stats"]})
            for pitcher in block["pitchers"]:
                pid = str(pitcher["player_id"])
                player_rows[pid] = pitcher.get("name")
                pitcher_rows.append({
                    "game_id": game.game_id, "team": team, "player_id": pid,
                    "seq": pitcher["seq"], "is_start": bool(pitcher["is_start"]),
                    **{k: pitcher.get(k) for k in PITCHER.values()},
                })
    for side in ("home", "away"):
        ids, names = games.get(f"{side}_sp_id"), games.get(f"{side}_sp_name")
        if ids is not None:
            for pid, name in zip(ids, names):
                if pd.notna(pid):
                    player_rows.setdefault(str(int(pid)), name)
    players = pd.DataFrame({"player_id": list(player_rows), "name": list(player_rows.values())})
    players["throws"] = ""
    return {
        "games": games,
        "team_games": pd.DataFrame(team_rows),
        "pitcher_games": pd.DataFrame(pitcher_rows),
        "players": players,
    }
