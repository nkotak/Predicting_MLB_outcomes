"""Injured list moves from MLB's public transactions feed.

Each season's moves are reduced to the ones that put a player on the injured
list (called the disabled list before 2019) or take him off it. From those,
``spells`` works out the stretches of days each player was unavailable.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path

import pandas as pd

from .. import teams
from ..config import ensure, raw_dir
from ..http import get_json

log = logging.getLogger(__name__)

URL = "https://statsapi.mlb.com/api/v1/transactions?sportId=1&startDate={start}&endDate={end}"
LIST = r"(?:injured|disabled) list"
ON = re.compile(rf"\b(?:placed|transferred)\b.*\b(?:on|to) the (\d+)-day {LIST}", re.IGNORECASE)
OFF = re.compile(rf"\b(?:activated|reinstated|returned|recalled)\b.*\bfrom the (\d+)-day {LIST}", re.IGNORECASE)
# Moves that end a club's claim on a player whatever list he was on.
GONE = re.compile(r"\b(released|traded|claimed|selected off waivers|elected free agency|retired)\b", re.IGNORECASE)


def _file(season: int) -> Path:
    return ensure(raw_dir() / "mlb") / f"injured_list_{season}.json"


def slim(payload: dict) -> list[dict]:
    moves = []
    for item in payload.get("transactions", []):
        text = item.get("description") or ""
        person = (item.get("person") or {}).get("id")
        if person is None or not text:
            continue
        on, off = ON.search(text), OFF.search(text)
        if on and not (off and off.start() < on.start()):
            kind, days = "on", int(on.group(1))
        elif off:
            kind, days = "off", int(off.group(1))
        elif GONE.search(text):
            kind, days = "gone", 0
        else:
            continue
        team = (item.get("toTeam") or {}).get("id") if kind != "gone" else (item.get("fromTeam") or {}).get("id")
        moves.append({"date": item.get("date"), "player_id": int(person), "team_id": team, "kind": kind, "days": days})
    return moves


def fetch(season: int, *, refresh: bool = False) -> list[dict]:
    """Injured list moves for one season. A finished season is downloaded once."""
    path = _file(season)
    today = dt.date.today()
    finished = path.exists() and dt.datetime.fromtimestamp(path.stat().st_mtime) >= dt.datetime(season, 11, 15)
    if finished and not refresh:
        return json.loads(path.read_text())
    moves = []
    for month in range(2, 12):
        start = dt.date(season, month, 1)
        end = dt.date(season, month + 1, 1) - dt.timedelta(days=1)
        if start > today:
            break
        moves.extend(slim(get_json(URL.format(start=start.isoformat(), end=min(end, today).isoformat()))))
    path.write_text(json.dumps(moves))
    return moves


def load(season: int) -> list[dict]:
    path = _file(season)
    return json.loads(path.read_text()) if path.exists() else []


def spells(seasons: list[int]) -> pd.DataFrame:
    """One row per stretch on the injured list: player, team, first day and last day.

    A stretch starts on the day the move was announced, so nothing here was
    unknown at the time. It ends when the player is activated, leaves the
    club, or the season ends.
    """
    rows = []
    for season in seasons:
        moves = load(season)
        if not moves:
            continue
        frame = pd.DataFrame(moves)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values(["player_id", "date"], kind="stable")
        season_end = pd.Timestamp(season, 11, 5)
        for player, group in frame.groupby("player_id", sort=False):
            start = team = None
            for move in group.itertuples(index=False):
                if move.kind == "on":
                    if start is None:
                        start, team = move.date, move.team_id
                elif start is not None:
                    rows.append((str(player), team, start, move.date - pd.Timedelta(days=1)))
                    start = team = None
            if start is not None:
                rows.append((str(player), team, start, season_end))
    out = pd.DataFrame(rows, columns=["player_id", "team_id", "start", "end"])
    if out.empty:
        out["team"] = pd.Series(dtype=object)
        return out
    known = {t.mlb_id for t in teams.TEAMS}
    out = out[out["team_id"].isin(known) & (out["end"] >= out["start"])].copy()
    out["team"] = out["team_id"].map(lambda value: teams.from_mlb_id(int(value)))
    out["player_id"] = out["player_id"].astype(object)
    return out[["player_id", "team", "start", "end"]].reset_index(drop=True)
