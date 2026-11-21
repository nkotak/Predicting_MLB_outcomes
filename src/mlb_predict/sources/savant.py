"""Season tables from MLB's Statcast site, Baseball Savant.

Three public season leaderboards are stored per season:

* expected hitting: how well each hitter and pitcher "should" have done given
  how hard and at what angle balls were hit (xwOBA), next to what happened
* outs above average: how many outs each fielder saved or cost
* sprint speed

These are whole-season numbers, so a game is only ever given the numbers from
the season before it.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
from pathlib import Path

import pandas as pd

from ..config import ensure, raw_dir
from ..http import download

log = logging.getLogger(__name__)

FIRST_SEASON = 2015
BASE = "https://baseballsavant.mlb.com/leaderboard/"
PAGES = {
    "expected_batter": "expected_statistics?type=batter&year={season}&position=&team=&min=25&csv=true",
    "expected_pitcher": "expected_statistics?type=pitcher&year={season}&position=&team=&min=25&csv=true",
    "outs_above_average": ("outs_above_average?type=Fielder&startYear={season}&endYear={season}&split=no&team="
                           "&range=year&min=10&pos=&roles=&viz=hide&csv=true"),
    "sprint_speed": "sprint_speed?min_season={season}&max_season={season}&position=&team=&min=10&csv=true",
}


def _file(page: str, season: int) -> Path:
    return ensure(raw_dir() / "savant") / f"{page}_{season}.csv"


def written_before_season_end(path: Path, season: int) -> bool:
    """True if a stored season file was saved while that season was still being played."""
    if not path.exists():
        return True
    return dt.datetime.fromtimestamp(path.stat().st_mtime) < dt.datetime(season, 11, 15)


def fetch(season: int, *, refresh: bool = False) -> int:
    """Download one season's tables. Returns how many were fetched or already stored.

    A table saved while its season was still in progress is downloaded again,
    so that the finished season's numbers are the ones kept.
    """
    got = 0
    for page, path in PAGES.items():
        target = _file(page, season)
        again = refresh or (target.exists() and written_before_season_end(target, season))
        try:
            got += bool(download(BASE + path.format(season=season), target, refresh=again, allow_404=True))
        except Exception as exc:  # one missing leaderboard must not stop the whole download
            log.warning("Savant %s %d not downloaded: %s", page, season, exc)
    return got


def _read(page: str, season: int) -> pd.DataFrame | None:
    path = _file(page, season)
    if not path.exists() or path.stat().st_size < 50:
        return None
    text = path.read_bytes().decode("utf-8-sig", errors="replace")
    if text.lstrip().startswith("<"):  # an error page, not a table
        return None
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [c.strip().strip('"') for c in frame.columns]
    return frame if "player_id" in frame.columns else None


def load(seasons: list[int]) -> dict[str, pd.DataFrame]:
    """Stored tables for ``seasons`` in a tidy form, with text player ids.

    ``expected``  player_id, season, role (bat or pit), pa, woba, xwoba
    ``fielding``  player_id, season, oaa
    ``speed``     player_id, season, sprint_speed
    """
    expected, fielding, speed = [], [], []
    for season in seasons:
        for role, page in (("bat", "expected_batter"), ("pit", "expected_pitcher")):
            frame = _read(page, season)
            if frame is not None and {"pa", "woba", "est_woba"} <= set(frame.columns):
                expected.append(pd.DataFrame({
                    "player_id": frame["player_id"].astype("int64").astype(str), "season": season, "role": role,
                    "pa": pd.to_numeric(frame["pa"], errors="coerce"),
                    "woba": pd.to_numeric(frame["woba"], errors="coerce"),
                    "xwoba": pd.to_numeric(frame["est_woba"], errors="coerce")}))
        frame = _read("outs_above_average", season)
        if frame is not None and "outs_above_average" in frame.columns:
            fielding.append(pd.DataFrame({
                "player_id": frame["player_id"].astype("int64").astype(str), "season": season,
                "oaa": pd.to_numeric(frame["outs_above_average"], errors="coerce")}))
        frame = _read("sprint_speed", season)
        if frame is not None and "sprint_speed" in frame.columns:
            speed.append(pd.DataFrame({
                "player_id": frame["player_id"].astype("int64").astype(str), "season": season,
                "sprint_speed": pd.to_numeric(frame["sprint_speed"], errors="coerce")}))

    def stack(parts, columns):
        if not parts:
            return pd.DataFrame(columns=columns)
        out = pd.concat(parts, ignore_index=True)
        out["player_id"] = out["player_id"].astype(object)
        return out

    return {"expected": stack(expected, ["player_id", "season", "role", "pa", "woba", "xwoba"]),
            "fielding": stack(fielding, ["player_id", "season", "oaa"]).groupby(
                ["player_id", "season"], as_index=False)["oaa"].sum() if fielding else stack([], ["player_id", "season", "oaa"]),
            "speed": stack(speed, ["player_id", "season", "sprint_speed"])}
