"""Downloading: the first full download and the small daily refresh."""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from . import build
from .config import FIRST_ODDS_SEASON, FIRST_SEASON
from .sources import crosswalk, espn, gamefeed, mlbapi, retrosheet, savant, transactions

log = logging.getLogger(__name__)


def _mlb_seasons(today: dt.date, first: int) -> list[int]:
    """Seasons that have to come from MLB's feed: not on Retrosheet yet, plus next year."""
    return [s for s in range(first, today.year + 2) if not retrosheet.zip_path(s).exists()]


def fetch_all(*, first_season: int = FIRST_SEASON, odds: bool = True, refresh: bool = False) -> dict:
    """Download everything needed from scratch. Safe to stop and run again."""
    today = dt.date.today()
    published = [s for s in range(first_season, today.year + 1) if retrosheet.fetch(s, refresh=refresh)]
    log.info("Retrosheet seasons available: %s to %s", published[0] if published else None,
             published[-1] if published else None)
    crosswalk.fetch(refresh=refresh)
    from_mlb = []
    for season in _mlb_seasons(today, first_season):
        games = mlbapi.fetch_schedule(season, refresh=True)
        if not games:
            continue
        from_mlb.append(season)
        mlbapi.fetch_boxscores(season)
    summary = {"retrosheet_seasons": published, "mlb_seasons": from_mlb, "odds_requests": 0}
    summary.update(fetch_player_detail(range(first_season, today.year + 1)))
    if odds:
        games = build.combine(first_season)["games"]
        dates = [d.date() for d in pd.to_datetime(games.loc[games["season"] >= FIRST_ODDS_SEASON, "date"].unique())]
        past = [d for d in dates if d <= today + dt.timedelta(days=3)]
        summary["odds_requests"] = espn.fetch_odds(past, refresh_from=today - dt.timedelta(days=1))
    return summary


def fetch_player_detail(seasons) -> dict:
    """Lineups, plate appearances and rosters for every finished game, plus injuries and Statcast tables.

    One request per game, so the first run takes a while (about five minutes a
    season). Safe to stop and run again: games already stored are skipped.
    """
    downloaded = failed = 0
    for season in seasons:
        if not mlbapi.fetch_schedule(season):
            continue
        result = gamefeed.fetch(season)
        downloaded += result["downloaded"]
        failed += result["failed"]
        transactions.fetch(season)
        if season >= savant.FIRST_SEASON:
            savant.fetch(season)
    if failed:
        log.warning("%d games could not be downloaded this time. Run `mlb fetch` again to fill them in.", failed)
    return {"game_feeds_downloaded": downloaded, "game_feeds_failed": failed}


def fetch_upcoming(days_ahead: int = 3) -> int:
    """Rosters and, where posted, starting lineups for games in the next few days."""
    today = dt.date.today()
    pks = []
    for season in {today.year, (today + dt.timedelta(days=days_ahead)).year}:
        path = mlbapi._dir() / f"schedule_{season}.json"
        if not path.exists():
            continue
        for game in mlbapi.fetch_schedule(season):
            if game["state"] in mlbapi.FINAL_STATES or game["state"] in mlbapi.DEAD_STATES:
                continue
            date = dt.date.fromisoformat(game["official_date"])
            if today - dt.timedelta(days=1) <= date <= today + dt.timedelta(days=days_ahead):
                pks.append(int(game["game_pk"]))
    return gamefeed.fetch_pregame(pks)


def update(*, days_back: int = 4, days_ahead: int = 3) -> dict:
    """Daily refresh: new results, current starters, current prices. Then rebuild.

    Only the season in progress and the next few days are downloaded again.
    """
    today = dt.date.today()
    seasons = [s for s in (today.year, today.year + 1) if not retrosheet.zip_path(s).exists()]
    new_boxes = 0
    for season in seasons:
        if mlbapi.fetch_schedule(season, refresh=True):
            new_boxes += mlbapi.fetch_boxscores(season)
    detail = fetch_player_detail([s for s in seasons if s <= today.year])
    upcoming = fetch_upcoming(days_ahead)
    window = [today + dt.timedelta(days=offset) for offset in range(-days_back, days_ahead + 1)]
    # Every day in the window is downloaded again: upcoming games because their
    # prices keep moving, recent games so the stored price is the closing one.
    requests = espn.fetch_odds(window, refresh_from=window[0])
    report = build.run()
    return {"new_box_scores": new_boxes, "odds_requests": requests, "build": report,
            "new_game_feeds": detail["game_feeds_downloaded"], "upcoming_games_checked": upcoming}
