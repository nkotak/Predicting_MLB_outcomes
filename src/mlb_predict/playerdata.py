"""Player-level tables, built from MLB's game feed and lined up with ``games``.

Tables (all use the same ``game_id``, team codes and player ids as the rest of
the project):

    lineups             the nine starters of each team, in batting order
    rosters             everyone on the active roster that day, with a role
    plate_appearances   totals for each batter against each pitcher in a game
    game_extras         weather, home plate umpire and ballpark
    people              name, batting side, throwing hand, birth date
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import teams
from .config import ensure, processed_dir
from .sources import gamefeed, mlbapi, savant, transactions

log = logging.getLogger(__name__)

FEED_TABLES = ("lineups", "rosters", "plate_appearances", "game_extras", "people")
EXTRA_TABLES = ("il_spells", "sc_expected", "sc_fielding", "sc_speed")
TABLES = FEED_TABLES + EXTRA_TABLES
MIN_GAME_MATCH = 0.99
MIN_STARTER_AGREEMENT = 0.99


def _cache_dir():
    return ensure(processed_dir() / "feed_cache")


def _season_tables(season: int) -> dict[str, pd.DataFrame] | None:
    """Parsed tables for one season, reusing the last parse when the download has not changed."""
    raw = gamefeed._file(season)
    if not raw.exists():
        return None
    stamp = f"{raw.stat().st_size}"
    folder = _cache_dir()
    marker = folder / f"{season}.stamp"
    paths = {name: folder / f"{season}_{name}.parquet" for name in FEED_TABLES}
    if marker.exists() and marker.read_text() == stamp and all(p.exists() for p in paths.values()):
        return {name: pd.read_parquet(path) for name, path in paths.items()}
    tables = gamefeed.parse(season)
    for name, path in paths.items():
        tables[name].to_parquet(path, index=False)
    marker.write_text(stamp)
    return tables


def game_numbers(games: pd.DataFrame) -> pd.DataFrame:
    """MLB's game number (``pk``) for every game, as ``game_id`` and ``pk``.

    Games that came from MLB's feed already carry it. For seasons read from
    Retrosheet the MLB schedule is matched on date, home team and game number,
    then on date and teams, then one day either side for moved games, and
    last with home and away swapped.
    """
    known = games.loc[games["mlb_game_pk"].notna(), ["game_id", "mlb_game_pk"]].rename(columns={"mlb_game_pk": "pk"})
    parts = [known.astype({"pk": "int64"})]
    missing = games[games["mlb_game_pk"].isna()]
    for season, rows in missing.groupby("season"):
        path = mlbapi._dir() / f"schedule_{int(season)}.json"
        if not path.exists():
            continue
        listed = mlbapi.schedule_frame(mlbapi.fetch_schedule(int(season)))
        listed = listed[listed["status"] == "final"]
        if listed.empty:
            continue
        listed = listed[["game_id", "date", "home", "away", "number", "mlb_game_pk"]].rename(
            columns={"mlb_game_pk": "pk", "game_id": "listed_id"})
        exact = rows[["game_id"]].merge(listed, left_on="game_id", right_on="listed_id")[["game_id", "pk"]]
        parts.append(exact.astype({"pk": "int64"}))
        left = rows[~rows["game_id"].isin(exact["game_id"])][["game_id", "date", "home", "away", "number"]]
        free = listed[~listed["pk"].isin(exact["pk"])]
        # The last pass swaps home and away. A few games are played in the
        # "visiting" club's park (a shared stadium, a storm, the 2020 season):
        # Retrosheet then lists the park's club as home and MLB lists the club
        # that batted last.
        for shift, swapped in ((0, False), (-1, False), (1, False), (0, True)):
            if left.empty or free.empty:
                break
            a = left.sort_values(["date", "home", "away", "number"]).copy()
            a["slot"] = a.groupby(["date", "home", "away"]).cumcount()
            b = free.sort_values(["date", "home", "away", "number"]).copy()
            if swapped:
                b = b.rename(columns={"home": "away", "away": "home"}).sort_values(["date", "home", "away", "number"])
            b["date"] = b["date"] + pd.Timedelta(days=shift)
            b["slot"] = b.groupby(["date", "home", "away"]).cumcount()
            hit = a.merge(b[["date", "home", "away", "slot", "pk"]], on=["date", "home", "away", "slot"])
            hit = hit.drop_duplicates("game_id").drop_duplicates("pk")[["game_id", "pk"]]
            parts.append(hit.astype({"pk": "int64"}))
            left = left[~left["game_id"].isin(hit["game_id"])]
            free = free[~free["pk"].isin(hit["pk"])]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["game_id", "pk"])
    return out.drop_duplicates("game_id").drop_duplicates("pk").reset_index(drop=True)


def _team_codes(ids: pd.Series) -> pd.Series:
    lookup = {team_id: teams.from_mlb_id(team_id) for team_id in ids.dropna().unique()}
    return ids.map(lookup)


def _player_ids(ids: pd.Series) -> pd.Series:
    return ids.astype("int64").astype(str).astype(object)


def parse_wind(text: pd.Series) -> pd.DataFrame:
    """Wind speed in mph, and the part of it blowing out to (plus) or in from (minus) the outfield."""
    text = text.fillna("").astype(str)
    speed = pd.to_numeric(text.str.extract(r"(\d+)\s*mph", expand=False), errors="coerce").fillna(0.0)
    lowered = text.str.lower()
    direction = np.where(lowered.str.contains("out to"), 1.0, np.where(lowered.str.contains("in from"), -1.0, 0.0))
    return pd.DataFrame({"wind_mph": speed, "wind_out": speed * direction})


def assemble(games: pd.DataFrame, raw: list[dict[str, pd.DataFrame]], numbers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Attach ``game_id``, team codes and text player ids to tables keyed by ``pk``."""
    out: dict[str, pd.DataFrame] = {}
    key = numbers[["pk", "game_id"]]

    def stack(name: str) -> pd.DataFrame:
        frames = [t[name] for t in raw if len(t[name])]
        return pd.concat(frames, ignore_index=True) if frames else raw[0][name].iloc[0:0] if raw else pd.DataFrame()

    lineups = stack("lineups").merge(key, on="pk")
    lineups["team"] = _team_codes(lineups["team_id"])
    lineups["player_id"] = _player_ids(lineups["player_id"])
    out["lineups"] = lineups[["game_id", "team", "slot", "player_id", "position"]].drop_duplicates(
        ["game_id", "team", "slot"], keep="last").reset_index(drop=True)

    rosters = stack("rosters").merge(key, on="pk")
    rosters["team"] = _team_codes(rosters["team_id"])
    rosters["player_id"] = _player_ids(rosters["player_id"])
    out["rosters"] = rosters[["game_id", "team", "player_id", "role"]].drop_duplicates(
        ["game_id", "team", "player_id"], keep="last").reset_index(drop=True)

    pa = stack("plate_appearances").merge(key, on="pk")
    pa["bat_team"] = _team_codes(pa["bat_team_id"])
    pa["pit_team"] = _team_codes(pa["pit_team_id"])
    pa["batter"] = _player_ids(pa["batter"])
    pa["pitcher"] = _player_ids(pa["pitcher"])
    keep = ["game_id", "bat_team", "pit_team", "batter", "pitcher", "bat_side", "pit_hand", "vs_starter",
            *gamefeed.PA_SUMS]
    out["plate_appearances"] = pa[keep].reset_index(drop=True)

    extras = stack("game_extras").merge(key, on="pk").drop_duplicates("game_id", keep="last")
    wind = parse_wind(extras["wind"]).set_index(extras.index)
    extras = pd.concat([extras, wind], axis=1)
    extras["temp"] = pd.to_numeric(extras["temp"], errors="coerce")
    extras["venue"] = pd.to_numeric(extras["venue"], errors="coerce")
    extras["umpire"] = pd.to_numeric(extras["umpire"], errors="coerce")
    sky = extras["sky"].fillna("").str.lower()
    extras["roof_closed"] = sky.str.contains("roof closed|dome").astype(float)
    out["game_extras"] = extras[["game_id", "venue", "temp", "wind_mph", "wind_out", "roof_closed",
                                 "sky", "umpire"]].reset_index(drop=True)

    people = stack("people")
    if len(people):
        people = people.drop_duplicates("player_id", keep="last")
        people["player_id"] = _player_ids(people["player_id"])
    out["people"] = people.reset_index(drop=True)
    return out


def build(games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every player-level table for the games in ``games``."""
    numbers = game_numbers(games)
    raw = []
    for season in sorted(int(s) for s in games["season"].unique()):
        tables = _season_tables(season)
        if tables is not None:
            raw.append(tables)
    pregame = gamefeed.load_pregame()
    if pregame:
        scheduled = set(numbers.loc[numbers["game_id"].isin(games.loc[games["status"] == "scheduled", "game_id"]), "pk"])
        upcoming = [r for r in pregame if int(r["pk"]) in scheduled]
        if upcoming:
            raw.append(gamefeed.tables_from(upcoming))
    if not raw:
        raise FileNotFoundError("no game feed downloads found. Run `mlb fetch` first.")
    tables = assemble(games, raw, numbers)
    seasons = sorted(int(s) for s in games["season"].unique())
    tables["il_spells"] = transactions.spells(seasons)
    statcast = savant.load(seasons)
    tables["sc_expected"], tables["sc_fielding"], tables["sc_speed"] = (
        statcast["expected"], statcast["fielding"], statcast["speed"])
    return tables


def check(games: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> tuple[dict, list[str]]:
    """Sanity checks on the player tables. Returns a report and a list of problems."""
    problems: list[str] = []
    report: dict = {}
    final = games[games["status"] == "final"]
    lineups, pa = tables["lineups"], tables["plate_appearances"]

    per_side = lineups.groupby(["game_id", "team"]).size()
    full = per_side[per_side == 9].reset_index()[["game_id", "team"]]
    sides = pd.concat([final[["game_id", "home"]].rename(columns={"home": "team"}),
                       final[["game_id", "away"]].rename(columns={"away": "team"})])
    covered = sides.merge(full.assign(has=1.0), on=["game_id", "team"], how="left")["has"].fillna(0.0) > 0
    by_season = covered.groupby(sides["game_id"].map(final.set_index("game_id")["season"]).to_numpy()).mean()
    report["lineup_coverage"] = {int(k): round(float(v), 4) for k, v in by_season.items()}
    if len(covered) and covered.mean() < MIN_GAME_MATCH:
        problems.append(f"only {covered.mean():.1%} of finished team-games have a nine-man starting lineup "
                        f"(need {MIN_GAME_MATCH:.0%}). Run `mlb fetch` again to fill in missing games.")

    # The pitcher the feed says faced the first batter should be the starter the box score names.
    first = pa[pa["vs_starter"]].drop_duplicates(["game_id", "pit_team"])[["game_id", "pit_team", "pitcher"]]
    named = pd.concat([final[["game_id", "home", "home_sp_id"]].set_axis(["game_id", "pit_team", "sp"], axis=1),
                       final[["game_id", "away", "away_sp_id"]].set_axis(["game_id", "pit_team", "sp"], axis=1)])
    both = first.merge(named, on=["game_id", "pit_team"])
    if len(both):
        agree = float((both["pitcher"].astype(str) == both["sp"].astype(str)).mean())
        report["starter_agreement"] = round(agree, 5)
        if agree < MIN_STARTER_AGREEMENT:
            problems.append(f"the game feed and the box scores name the same starting pitcher in only {agree:.1%} of games")

    # Plate appearances per finished game should look like baseball.
    per_game = pa.groupby("game_id")["pa"].sum()
    typical = per_game[per_game.index.isin(final["game_id"])]
    if len(typical):
        report["plate_appearances_per_game"] = round(float(typical.mean()), 2)
        if not 70 <= typical.mean() <= 82:
            problems.append(f"average plate appearances per game is {typical.mean():.1f}, outside 70 to 82")
    return report, problems
