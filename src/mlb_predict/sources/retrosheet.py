"""Game history from Retrosheet's per-season CSV files.

Each season is one zip holding a row per game, a row per team per game and a
row per pitcher per game. Players are identified by Retrosheet's own ids, so
no matching by name is needed.

The information used here was obtained free of charge from and is copyrighted
by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

from .. import teams
from ..config import ensure, raw_dir
from ..http import download

log = logging.getLogger(__name__)

URL = "https://www.retrosheet.org/downloads/{season}/{season}csvs.zip"

TEAM_COLUMNS = {
    "b_pa": "pa", "b_ab": "ab", "b_r": "runs", "b_h": "h", "b_d": "doubles", "b_t": "triples",
    "b_hr": "hr", "b_w": "bb", "b_iw": "ibb", "b_hbp": "hbp", "b_k": "so", "b_sf": "sf",
    "b_sb": "sb", "b_cs": "cs", "lob": "lob",
    "p_ipouts": "p_outs", "p_bfp": "p_bf", "p_h": "p_h", "p_hr": "p_hr", "p_r": "p_r",
    "p_er": "p_er", "p_w": "p_bb", "p_k": "p_so", "p_hbp": "p_hbp", "d_e": "errors",
}
PITCHER_COLUMNS = {
    "p_ipouts": "outs", "p_bfp": "bf", "p_h": "h", "p_hr": "hr", "p_r": "r", "p_er": "er",
    "p_w": "bb", "p_iw": "ibb", "p_k": "so", "p_hbp": "hbp",
}


def zip_path(season: int) -> Path:
    return ensure(raw_dir() / "retrosheet") / f"{season}csvs.zip"


def fetch(season: int, *, refresh: bool = False) -> bool:
    """Download one season. Returns False if Retrosheet has not published it yet."""
    return download(URL.format(season=season), zip_path(season), refresh=refresh, allow_404=True)


def _read(archive: zipfile.ZipFile, season: int, table: str, usecols: list[str]) -> pd.DataFrame:
    wanted = f"{season}{table}.csv"
    member = next((n for n in archive.namelist() if n.lower().endswith(wanted)), None)
    if member is None:
        raise FileNotFoundError(f"{wanted} is missing from the Retrosheet {season} download")
    with archive.open(member) as handle:
        return pd.read_csv(handle, usecols=usecols, dtype=str, keep_default_na=False)


def _game_id(date: pd.Series, home: pd.Series, number: pd.Series) -> pd.Series:
    return date.dt.strftime("%Y%m%d") + "-" + home + "-" + number.astype(str)


def parse(season: int) -> dict[str, pd.DataFrame]:
    """Read one season into the project's standard tables.

    Returns ``games``, ``team_games``, ``pitcher_games`` and ``players``.
    Player ids are Retrosheet ids here. They are converted later, in the build.
    """
    with zipfile.ZipFile(zip_path(season)) as archive:
        info = _read(archive, season, "gameinfo",
                     ["gid", "visteam", "hometeam", "site", "date", "number", "daynight",
                      "gametype", "vruns", "hruns"])
        stats = _read(archive, season, "teamstats",
                      ["gid", "team", "stattype", "gametype", "vishome", *TEAM_COLUMNS])
        pitch = _read(archive, season, "pitching",
                      ["gid", "id", "team", "p_seq", "stattype", "gametype", "p_gs", *PITCHER_COLUMNS])
        people = _read(archive, season, "allplayers", ["id", "last", "first", "throw"])

    info = info[info["gametype"] == "regular"].copy()
    info["date"] = pd.to_datetime(info["date"], format="%Y%m%d")
    info["home"] = info["hometeam"].map(teams.from_retro)
    info["away"] = info["visteam"].map(teams.from_retro)
    info["number"] = pd.to_numeric(info["number"]).astype(int)
    info["game_id"] = _game_id(info["date"], info["home"], info["number"])
    gid_to_game = dict(zip(info["gid"], info["game_id"]))

    games = pd.DataFrame({
        "game_id": info["game_id"],
        "date": info["date"],
        "season": season,
        "number": info["number"],
        "home": info["home"],
        "away": info["away"],
        "home_score": pd.to_numeric(info["hruns"], errors="coerce"),
        "away_score": pd.to_numeric(info["vruns"], errors="coerce"),
        "status": "final",
        "day_night": info["daynight"].str.lower().str[:1].map({"d": "D", "n": "N"}),
        "park": info["site"],
        "source": "retrosheet",
        "mlb_game_pk": pd.NA,
    }).reset_index(drop=True)

    stats = stats[(stats["gametype"] == "regular") & (stats["stattype"] == "value")].copy()
    team_games = pd.DataFrame({
        "game_id": stats["gid"].map(gid_to_game),
        "team": stats["team"].map(teams.from_retro),
        "is_home": stats["vishome"].str.lower().eq("h"),
    })
    for source, target in TEAM_COLUMNS.items():
        team_games[target] = pd.to_numeric(stats[source], errors="coerce")
    team_games = team_games.reset_index(drop=True)

    pitch = pitch[(pitch["gametype"] == "regular") & (pitch["stattype"] == "value")].copy()
    pitcher_games = pd.DataFrame({
        "game_id": pitch["gid"].map(gid_to_game),
        "team": pitch["team"].map(teams.from_retro),
        "player_id": pitch["id"],
        "seq": pd.to_numeric(pitch["p_seq"], errors="coerce"),
        "is_start": pitch["p_gs"].eq("1"),
    })
    for source, target in PITCHER_COLUMNS.items():
        pitcher_games[target] = pd.to_numeric(pitch[source], errors="coerce")
    pitcher_games = pitcher_games.reset_index(drop=True)

    players = (
        people.assign(name=(people["first"].str.strip() + " " + people["last"].str.strip()).str.strip())
        .rename(columns={"id": "player_id", "throw": "throws"})[["player_id", "name", "throws"]]
        .drop_duplicates("player_id")
        .reset_index(drop=True)
    )
    return {"games": games, "team_games": team_games, "pitcher_games": pitcher_games, "players": players}
