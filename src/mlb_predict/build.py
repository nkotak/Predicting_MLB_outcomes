"""Turn raw downloads into the project's clean tables, and check them.

Tables written to ``data/processed``:

    games          one row per game (finished or scheduled)
    team_games     two rows per finished game: each team's batting and pitching totals
    pitcher_games  one row per pitcher appearance
    players        id and name for every pitcher
    odds           closing (or current) moneyline per game, where available

plus the player-level tables described in ``playerdata``: lineups, rosters,
plate appearances, game extras, people, injured list stretches and Statcast
season tables.

Every check that fails stops the build with a message that says what is wrong.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from . import playerdata, teams
from .config import FIRST_SEASON, ensure, processed_dir
from .sources import crosswalk, espn, mlbapi, retrosheet

log = logging.getLogger(__name__)

TABLES = ("games", "team_games", "pitcher_games", "players", "odds")
#: Player-level tables from MLB's game feed. See ``playerdata``.
PLAYER_TABLES = playerdata.TABLES
ALL_TABLES = TABLES + PLAYER_TABLES
# Favourites at pre-game prices win about 57 to 59 percent of MLB games. A
# season well above that, on a large sample, means in-game prices got in.
MIN_GAMES_FOR_MARKET_CHECK = 1500
MAX_FAVOURITE_WIN_RATE = 0.615


class DataCheckError(RuntimeError):
    """The built data failed a sanity check."""


# --------------------------------------------------------------------------
# Loading and saving
# --------------------------------------------------------------------------


def save(tables: dict[str, pd.DataFrame]) -> None:
    out = ensure(processed_dir())
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)


def load(names: tuple[str, ...] = TABLES) -> dict[str, pd.DataFrame]:
    out = processed_dir()
    missing = [n for n in names if not (out / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing tables {missing} in {out}. Run `mlb fetch` and then `mlb build` first."
        )
    return {name: pd.read_parquet(out / f"{name}.parquet") for name in names}


def load_all() -> dict[str, pd.DataFrame]:
    """Every table, the player-level ones included."""
    return load(ALL_TABLES)


# --------------------------------------------------------------------------
# Combining the sources
# --------------------------------------------------------------------------


def _season_source(season: int) -> str | None:
    if retrosheet.zip_path(season).exists():
        return "retrosheet"
    if (mlbapi._dir() / f"schedule_{season}.json").exists():
        return "mlbapi"
    return None


def _attach_starters(games: pd.DataFrame, pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """Set each finished game's starters from who actually started."""
    starts = pitcher_games[pitcher_games["is_start"]]
    games = games.copy()
    for column in ("home_sp_id", "away_sp_id"):
        if column not in games.columns:
            games[column] = None
        games[column] = games[column].astype(object)
    for side, column in (("home", "home_sp_id"), ("away", "away_sp_id")):
        lookup = (
            starts.merge(games[["game_id", side]], left_on=["game_id", "team"], right_on=["game_id", side])
            .drop_duplicates("game_id")
            .set_index("game_id")["player_id"]
        )
        actual = games["game_id"].map(lookup)
        games[column] = actual.where(actual.notna(), games[column])
    return games


def combine(first_season: int = FIRST_SEASON, last_season: int | None = None) -> dict[str, pd.DataFrame]:
    """Read every available season into one set of tables with one id scheme."""
    last_season = last_season or dt.date.today().year + 1
    retro_map = crosswalk.retro_to_mlb()
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in ("games", "team_games", "pitcher_games", "players")}
    used: dict[int, str] = {}
    for season in range(first_season, last_season + 1):
        source = _season_source(season)
        if source is None:
            continue
        tables = retrosheet.parse(season) if source == "retrosheet" else mlbapi.parse(season)
        if tables["games"].empty:
            continue
        used[season] = source
        if source == "retrosheet":
            def to_canonical(ids: pd.Series) -> pd.Series:
                mapped = ids.map(retro_map)
                return mapped.where(mapped.notna(), "r:" + ids)
            tables["pitcher_games"]["player_id"] = to_canonical(tables["pitcher_games"]["player_id"])
            tables["players"]["player_id"] = to_canonical(tables["players"]["player_id"])
        else:
            for column in ("home_sp_id", "away_sp_id"):
                ids = tables["games"][column]
                tables["games"][column] = ids.map(lambda v: None if pd.isna(v) else str(int(v))).astype(object)
        if not tables["pitcher_games"].empty:
            tables["pitcher_games"]["player_id"] = tables["pitcher_games"]["player_id"].astype(str)
            tables["games"] = _attach_starters(tables["games"], tables["pitcher_games"])
        for key in parts:
            if not tables[key].empty:
                parts[key].append(tables[key])
    if not parts["games"]:
        raise FileNotFoundError("no raw season data found. Run `mlb fetch` first.")

    games = pd.concat(parts["games"], ignore_index=True)
    keep = ["game_id", "date", "season", "number", "home", "away", "home_score", "away_score",
            "status", "day_night", "park", "source", "mlb_game_pk", "home_sp_id", "away_sp_id"]
    games = games[keep].copy()
    games["date"] = pd.to_datetime(games["date"])
    for column in ("home_score", "away_score", "mlb_game_pk"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games["home_sp_id"] = games["home_sp_id"].astype(object).where(games["home_sp_id"].notna(), None)
    games["away_sp_id"] = games["away_sp_id"].astype(object).where(games["away_sp_id"].notna(), None)
    games["park"] = games["park"].astype(str)
    games = games.sort_values(["date", "number", "home"]).reset_index(drop=True)

    team_games = pd.concat(parts["team_games"], ignore_index=True)
    pitcher_games = pd.concat(parts["pitcher_games"], ignore_index=True)
    numeric = [c for c in team_games.columns if c not in ("game_id", "team", "is_home")]
    team_games[numeric] = team_games[numeric].apply(pd.to_numeric, errors="coerce")
    numeric = [c for c in pitcher_games.columns if c not in ("game_id", "team", "player_id", "is_start")]
    pitcher_games[numeric] = pitcher_games[numeric].apply(pd.to_numeric, errors="coerce")
    pitcher_games["is_start"] = pitcher_games["is_start"].astype(bool)
    team_games["is_home"] = team_games["is_home"].astype(bool)

    players = pd.concat(parts["players"], ignore_index=True)
    players["player_id"] = players["player_id"].astype(str)
    players = players.sort_values("name").drop_duplicates("player_id", keep="last").reset_index(drop=True)
    players["throws"] = players["throws"].fillna("")

    log.info("seasons by source: %s", used)
    return {"games": games, "team_games": team_games, "pitcher_games": pitcher_games, "players": players}


# --------------------------------------------------------------------------
# Odds
# --------------------------------------------------------------------------


def match_odds(games: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach each ESPN event to one game.

    Games are matched on date, home team and away team. When the same two
    teams play twice on one date, the earlier ESPN start time goes with the
    lower game number. An event that finds no game on its own date is tried
    against the day before and after, which covers suspended and moved games.
    """
    columns = ["game_id", "event_id", "home_dec", "away_dec", "market_home_prob", "n_books",
               "best_home_dec", "best_away_dec",
               "open_home_dec", "open_away_dec", "open_market_home_prob",
               "open_best_home_dec", "open_best_away_dec", "odds_fetched_utc"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    events = events.copy()
    for column in columns[2:]:
        if column not in events.columns:
            events[column] = np.nan
    events = events.sort_values(["date", "home", "away", "start_utc"])
    events["slot"] = events.groupby(["date", "home", "away"]).cumcount()
    g = games[["game_id", "date", "home", "away", "number"]].sort_values(["date", "home", "away", "number"]).copy()
    g["slot"] = g.groupby(["date", "home", "away"]).cumcount()

    matched = events.merge(g, on=["date", "home", "away", "slot"], how="left")
    out = [matched[matched["game_id"].notna()]]
    loose = matched[matched["game_id"].isna()].drop(columns=["game_id", "number"])
    taken = set(out[0]["game_id"])
    if not loose.empty:
        for shift in (-1, 1):
            if loose.empty:
                break
            free = g[~g["game_id"].isin(taken)].copy()
            free["date"] = free["date"] + pd.Timedelta(days=shift)
            attempt = loose.merge(free.drop_duplicates(["date", "home", "away"]),
                                  on=["date", "home", "away"], how="left", suffixes=("", "_g"))
            hit = attempt[attempt["game_id"].notna()].drop_duplicates("game_id")
            out.append(hit)
            taken |= set(hit["game_id"])
            loose = loose[~loose["event_id"].isin(hit["event_id"])]
    result = pd.concat(out, ignore_index=True)
    result = result[result["home_dec"].notna()].drop_duplicates("game_id")
    return result[columns].reset_index(drop=True)


def build_odds(games: pd.DataFrame) -> pd.DataFrame:
    dates = [d.date() for d in pd.to_datetime(games["date"].unique())]
    events = espn.events_frame(dates)
    return match_odds(games, events)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check(tables: dict[str, pd.DataFrame]) -> dict:
    """Validate the built tables. Raises ``DataCheckError`` on any failure."""
    games, team_games, pitcher_games = tables["games"], tables["team_games"], tables["pitcher_games"]
    problems: list[str] = []
    report: dict = {}

    if games["game_id"].duplicated().any():
        dup = games.loc[games["game_id"].duplicated(), "game_id"].head(5).tolist()
        problems.append(f"duplicate game ids, for example {dup}")
    unknown = set(games["home"]) | set(games["away"])
    unknown -= set(teams.CODES)
    if unknown:
        problems.append(f"unknown team codes {sorted(unknown)}")

    final = games[games["status"] == "final"]
    per_game = team_games.groupby("game_id").size()
    bad = final.loc[final["game_id"].map(per_game).fillna(0) != 2, "game_id"]
    if len(bad):
        problems.append(f"{len(bad)} finished games do not have exactly two team rows, for example {bad.head(3).tolist()}")

    scores = team_games.merge(final[["game_id", "home_score", "away_score"]], on="game_id")
    expected = np.where(scores["is_home"], scores["home_score"], scores["away_score"])
    mismatch = scores[scores["runs"] != expected]
    if len(mismatch):
        problems.append(f"{len(mismatch)} team rows where runs do not match the final score")

    starts = pitcher_games[pitcher_games["is_start"]].groupby(["game_id", "team"]).size()
    if (starts != 1).any():
        problems.append(f"{int((starts != 1).sum())} team-games with more than one starting pitcher")
    missing_sp = final[final["home_sp_id"].isna() | final["away_sp_id"].isna()]
    if len(missing_sp) > 0.001 * len(final):
        problems.append(f"{len(missing_sp)} finished games have no starting pitcher")

    by_season = final.groupby("season").size()
    report["games_per_season"] = {int(k): int(v) for k, v in by_season.items()}
    this_year = dt.date.today().year
    for season, count in by_season.items():
        if season == 2020:
            ok = 880 <= count <= 910
        elif season >= this_year and count < 2400:
            ok = True  # season still in progress
        else:
            ok = 2400 <= count <= 2470
        if not ok:
            problems.append(f"season {season} has {count} finished games, outside the expected range")
        n_teams = len(set(final.loc[final.season == season, "home"]))
        if n_teams != 30 and count > 400:
            problems.append(f"season {season} has {n_teams} home teams, expected 30")

    sp_ids = pd.concat([final["home_sp_id"], final["away_sp_id"]]).dropna().astype(str)
    unmatched = sp_ids.str.startswith("r:").mean() if len(sp_ids) else 0.0
    report["starter_id_match_rate"] = round(1.0 - float(unmatched), 5)
    if unmatched > 0.01:
        problems.append(f"{unmatched:.1%} of starting pitchers have no MLB id in the crosswalk (limit 1%)")

    if "odds" in tables:
        odds = tables["odds"]
        coverage = (
            final.assign(has=final["game_id"].isin(odds["game_id"])).groupby("season")["has"].mean().round(4)
        )
        report["odds_coverage"] = {int(k): float(v) for k, v in coverage.items()}
        if len(odds):
            bad_prob = odds[(odds["market_home_prob"] <= 0.05) | (odds["market_home_prob"] >= 0.95)]
            if len(bad_prob) > 0.002 * len(odds):
                problems.append(f"{len(bad_prob)} games have an implausible market probability")
            # Pre-game favourites win about 57 to 59 percent of MLB games. A
            # season far above that means the prices were recorded during the
            # game, which would make every betting result meaningless.
            priced = final.merge(odds[["game_id", "market_home_prob"]], on="game_id")
            priced = priced[priced["home_score"] != priced["away_score"]]
            fav_won = (priced["market_home_prob"] > 0.5) == (priced["home_score"] > priced["away_score"])
            rate = fav_won.groupby(priced["season"]).agg(["mean", "size"])
            report["market_favourite_win_rate"] = {int(k): round(float(v), 4) for k, v in rate["mean"].items()}
            for season, row in rate.iterrows():
                if row["size"] >= MIN_GAMES_FOR_MARKET_CHECK and row["mean"] > MAX_FAVOURITE_WIN_RATE:
                    problems.append(
                        f"season {season}: market favourites won {row['mean']:.1%}, too high for pre-game "
                        "prices. The odds for this season look like in-game prices."
                    )

    report["scheduled_games"] = int((games["status"] == "scheduled").sum())
    report["finished_games"] = int(len(final))
    if problems:
        raise DataCheckError("data checks failed:\n  - " + "\n  - ".join(problems))
    return report


def run(first_season: int = FIRST_SEASON) -> dict:
    """Build, check and save every table. Returns the check report."""
    tables = combine(first_season)
    tables["odds"] = build_odds(tables["games"])
    report = check(tables)
    tables.update(playerdata.build(tables["games"]))
    player_report, problems = playerdata.check(tables["games"], tables)
    if problems:
        raise DataCheckError("data checks failed:\n  - " + "\n  - ".join(problems))
    report.update(player_report)
    save(tables)
    (ensure(processed_dir()) / "build_report.json").write_text(json.dumps(report, indent=2))
    log.info("build complete: %d finished games, %d scheduled", report["finished_games"], report["scheduled_games"])
    return report
