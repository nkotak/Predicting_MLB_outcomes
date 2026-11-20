"""Betting odds from ESPN's public feed.

Two calls are used:

* the daily scoreboard, which lists that day's games and their ESPN ids
* the odds list for one game, which holds the moneyline from each sportsbook

For a finished game the stored moneyline is the last price before first pitch,
which is the "closing line". For a game that has not started it is the current
price. The feed has real moneylines from the 2014 season onward.

Only the few fields this project needs are kept on disk.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .. import teams
from ..config import ensure, raw_dir
from ..http import get_json

log = logging.getLogger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date}&limit=100"
ODDS_URL = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{eid}/competitions/{eid}/odds?limit=100"

_write_lock = threading.Lock()


def _dir() -> Path:
    return ensure(raw_dir() / "espn")


def _american(value) -> float | None:
    """Parse an American moneyline such as -160, "+135" or "EVEN"."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().upper().replace("−", "-")
        if text in ("EVEN", "EV", "PK"):
            return 100.0
        try:
            value = float(text)
        except ValueError:
            return None
    value = float(value)
    if abs(value) < 100:  # ESPN stores 0 when a book has no price
        return None
    return value


def _nested_american(side: dict, key: str) -> float | None:
    block = side.get(key) or {}
    money = block.get("moneyLine") or {}
    return _american(money.get("american") or money.get("alternateDisplayValue"))


def slim_scoreboard(payload: dict, date: str) -> list[dict]:
    """Keep the handful of scoreboard fields needed to match games."""
    events = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        home, away = sides.get("home"), sides.get("away")
        if not home or not away:
            continue
        season_type = (event.get("season") or {}).get("type")
        events.append(
            {
                "event_id": str(event["id"]),
                "query_date": date,
                "start_utc": event.get("date"),
                "season_type": season_type,  # 2 = regular season
                "status": ((comp.get("status") or {}).get("type") or {}).get("name"),
                "home_espn_id": int(home["team"]["id"]),
                "away_espn_id": int(away["team"]["id"]),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
            }
        )
    return events


def slim_odds(payload: dict) -> list[dict]:
    """Keep one small record per sportsbook."""
    books = []
    for item in payload.get("items", []):
        home, away = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
        books.append(
            {
                "book": (item.get("provider") or {}).get("name"),
                "home": _american(home.get("moneyLine")),
                "away": _american(away.get("moneyLine")),
                "home_close": _nested_american(home, "close"),
                "away_close": _nested_american(away, "close"),
                "home_open": _nested_american(home, "open"),
                "away_open": _nested_american(away, "open"),
            }
        )
    return books


def fetch_scoreboard(date: dt.date, *, refresh: bool = False) -> list[dict]:
    """Events for one date. Cached unless the date is today or later."""
    key = date.strftime("%Y%m%d")
    path = ensure(_dir() / "scoreboard" / key[:4]) / f"{key}.json"
    is_past = date < dt.date.today() - dt.timedelta(days=1)
    if path.exists() and is_past and not refresh:
        return json.loads(path.read_text())
    payload = get_json(SCOREBOARD_URL.format(date=key))
    events = slim_scoreboard(payload, key)
    path.write_text(json.dumps(events))
    return events


def _odds_file(season: int) -> Path:
    return ensure(_dir() / "odds") / f"odds_{season}.jsonl"


def load_odds_records(season: int) -> dict[str, dict]:
    """All stored odds for a season, keyed by ESPN event id (latest record wins)."""
    path = _odds_file(season)
    records: dict[str, dict] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records[record["event_id"]] = record
    return records


def _fetch_one(event_id: str, season: int) -> dict:
    payload = get_json(ODDS_URL.format(eid=event_id), allow_404=True)
    record = {
        "event_id": event_id,
        "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "books": slim_odds(payload) if payload else [],
    }
    with _write_lock, _odds_file(season).open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def fetch_odds(dates: list[dt.date], *, refresh_from: dt.date | None = None, workers: int = 8) -> int:
    """Download scoreboards and odds for ``dates``.

    Finished work is skipped, so this can be stopped and restarted. Odds for
    games on or after ``refresh_from`` are downloaded again, because prices for
    games that have not started keep moving.
    Returns the number of odds requests made.
    """
    dates = sorted(set(dates))
    todo: list[tuple[str, int]] = []
    done_cache: dict[int, dict[str, dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        boards = list(pool.map(fetch_scoreboard, dates))
    for date, events in zip(dates, boards):
        season = date.year
        if season not in done_cache:
            done_cache[season] = load_odds_records(season)
        for event in events:
            if event.get("season_type") not in (2, None):
                continue  # spring training and postseason
            fresh = refresh_from is not None and date >= refresh_from
            if event["event_id"] not in done_cache[season] or fresh:
                todo.append((event["event_id"], season))
    log.info("ESPN odds: %d dates, %d games still to download", len(dates), len(todo))
    if not todo:
        return 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(lambda args: _fetch_one(*args), todo):
            done += 1
            if done % 2000 == 0:
                log.info("ESPN odds: %d of %d", done, len(todo))
    return done


# --------------------------------------------------------------------------
# Turning stored records into one row of prices per game
# --------------------------------------------------------------------------


def american_to_decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


# Entries in ESPN's odds list that are not a sportsbook's pre-game price.
# "Live Odds" entries hold prices taken while the game was being played. Using
# them would hand the market the answer, so they are never read.
NOT_A_PREGAME_BOOK = ("live", "consensus", "numberfire", "teamrankings", "accuscore")


MAX_BOOK_DISAGREEMENT = 0.08  # in win-chance points


def is_pregame_book(name: str | None) -> bool:
    lowered = (name or "").lower()
    return bool(lowered) and not any(word in lowered for word in NOT_A_PREGAME_BOOK)


def _middle_price(prices: list[tuple[float, float]]) -> dict | None:
    """Median price across books, after dropping broken markets and outliers."""
    home_dec, away_dec, fair = [], [], []
    for home, away in prices:
        dh, da = american_to_decimal(home), american_to_decimal(away)
        total = 1.0 / dh + 1.0 / da
        if not 1.0 <= total <= 1.15:  # not a real two-way market
            continue
        home_dec.append(dh)
        away_dec.append(da)
        fair.append((1.0 / dh) / total)
    if not fair:
        return None
    # Pre-game books agree with each other to within a few points. A book far
    # from the rest is stale, mistyped or an in-game price under another name.
    fair_arr, home_arr, away_arr = np.array(fair), np.array(home_dec), np.array(away_dec)
    keep = np.abs(fair_arr - np.median(fair_arr)) <= MAX_BOOK_DISAGREEMENT
    if not keep.any():  # the books disagree too much to call anything a market price
        return None
    return {"home_dec": float(np.median(home_arr[keep])), "away_dec": float(np.median(away_arr[keep])),
            "market_home_prob": float(np.median(fair_arr[keep])), "n_books": int(keep.sum())}


def consensus(books: list[dict]) -> dict | None:
    """Middle-of-the-market price across sportsbooks for one game.

    The main price is the closing price where ESPN records one, otherwise the
    stored price (for a game that has not started, that is the current price).
    For each book the bookmaker's margin is removed to get a fair home win
    chance, and the consensus is the median across books.

    Where books also record an opening price, the same is done for it and
    returned under ``open_`` names. Opening prices exist for recent seasons only.
    """
    books = [b for b in books if is_pregame_book(b.get("book"))]
    closing, opening = [], []
    for book in books:
        home, away = book.get("home_close"), book.get("away_close")
        if home is None or away is None:
            home, away = book.get("home"), book.get("away")
        if home is not None and away is not None:
            closing.append((home, away))
        if book.get("home_open") is not None and book.get("away_open") is not None:
            opening.append((book["home_open"], book["away_open"]))
    price = _middle_price(closing)
    if price is None:
        return None
    early = _middle_price(opening)
    if early is not None:
        price.update({"open_home_dec": early["home_dec"], "open_away_dec": early["away_dec"],
                      "open_market_home_prob": early["market_home_prob"]})
    return price


def events_frame(dates: list[dt.date]) -> pd.DataFrame:
    """Stored scoreboard events for ``dates`` with team codes and consensus prices."""
    rows = []
    odds_cache: dict[int, dict[str, dict]] = {}
    for date in sorted(set(dates)):
        key = date.strftime("%Y%m%d")
        path = _dir() / "scoreboard" / key[:4] / f"{key}.json"
        if not path.exists():
            continue
        if date.year not in odds_cache:
            odds_cache[date.year] = load_odds_records(date.year)
        for event in json.loads(path.read_text()):
            if event.get("season_type") not in (2, None):
                continue
            if event.get("status") in ("STATUS_POSTPONED", "STATUS_CANCELED"):
                continue
            row = {
                "event_id": event["event_id"],
                "date": pd.Timestamp(date),
                "start_utc": pd.to_datetime(event.get("start_utc"), utc=True, errors="coerce"),
                "home": teams.from_espn_id(event["home_espn_id"]),
                "away": teams.from_espn_id(event["away_espn_id"]),
                "status": event.get("status"),
            }
            record = odds_cache[date.year].get(event["event_id"])
            if record:
                price = consensus(record["books"])
                if price:
                    row.update(price)
                    row["odds_fetched_utc"] = record.get("fetched_utc")
            rows.append(row)
    return pd.DataFrame(rows)
