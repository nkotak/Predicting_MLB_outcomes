"""Player-by-player model inputs, built so that a game can never see its own result.

The team view in ``features.py`` treats a club as one lump. This module looks
inside it. For every game it works out, from earlier games only:

* lineup: how good the nine hitters actually starting are, each projected from
  his own plate appearances, more recent seasons counting for more
* left/right: how each hitter does against the throwing hand of today's
  starter, and how that starter does against left- and right-handed hitters
* history: what each hitter has done against this exact pitcher before
* starter "stuff": swings and misses, fastball speed, ground balls, hard contact
* bullpen: who is on the roster, how good each reliever is, and who is tired
  from pitching on the previous days
* defense: how often the team turns a ball in play into an out
* setting: ballpark, temperature, wind and the home plate umpire
* expected runs: all of the above rolled into runs each team should score

Every rate is pulled toward the league average by an amount that shrinks as
the player's own record grows, so a hot week does not look like a new player.

Two ways to choose the lineup are supported. ``posted`` uses the lineup that
was announced for the game (known a few hours before first pitch). Where none
is known, and always in ``projected`` mode, the lineup is guessed: the nine
hitters who started most often in the team's last few games against a
starter of the same throwing hand.

Hitting numbers are corrected for the ballpark each plate appearance happened
in, so a hitter is not flattered by a hitter-friendly home park.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _season_weight, order_key

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Plate appearances a game usually gives each lineup slot.
SLOT_WEIGHTS = np.array([4.65, 4.55, 4.45, 4.33, 4.24, 4.13, 4.01, 3.90, 3.77])
# How much "league average" each player carries, in plate appearances (or
# pitches, or balls in play). Bigger means slower to believe a player is unusual.
K_BAT = 220.0
K_BAT_SPLIT = 1000.0
K_BAT_K, K_BAT_BB, K_BAT_ISO = 120.0, 200.0, 300.0
K_PIT = 350.0
K_PIT_SPLIT = 700.0
K_PIT_K, K_PIT_BB = 150.0, 250.0
K_HISTORY = 150.0
K_WHIFF = 500.0
K_GB = 150.0
K_HARD = 200.0
K_SPEED = 40.0
K_DEFENSE = 1500.0
K_BAT_HARD = 100.0
K_HOT = 120.0                # a hot or cold three weeks counts for this little
HOT_DAYS = 21
TYPICAL_AGE = 28.5
K_UMPIRE = 150.0
K_PARK = 120.0
K_STARTS = 5.0
#: A hitter with no record is assumed a little below league average.
UNKNOWN_HITTER = 0.96
PITCHER_HITTING = 0.15       # what pitchers hit, used when a season has too few of them batting
WOBA_SCALE = 1.2             # points of wOBA per run, per plate appearance
PA_PER_GAME = 38.0
BF_PER_GAME = 38.0
USUAL_WINDOW = 20
RECENT_LINEUPS = 7           # games looked back over when a lineup has to be guessed
RECENCY = 0.8                # each game further back counts for this much of the one after it

TEAM_PLAYER_STATS = (
    "lu_woba", "lu_hand", "lu_k", "lu_bb", "lu_iso", "lu_exp", "lu_vs_usual",
    "mu_woba", "mu_k", "hist", "hist_pa",
    "sp_woba", "sp_vs_lu", "sp_whiff", "sp_csw", "sp_gb", "sp_speed", "sp_speed_trend",
    "bp_woba", "bp_k", "bp_tired",
    "def_der",
    "lu_hard", "sp_hard", "lu_xgap", "sp_xgap", "lu_oaa", "lu_speed",
    "il_bat", "il_pit", "il_n",
    "lu_hot", "sp_hot", "lu_age", "sp_age",
    "xr",
)
GAME_PLAYER_STATS = ("park_rf", "temp", "wind_out", "ump_runs", "x_total")

#: Every player-level column this module produces for the model to choose from.
PLAYER_FEATURES: tuple[str, ...] = tuple(f"d_{s}" for s in TEAM_PLAYER_STATS) + GAME_PLAYER_STATS

#: The same columns in named groups, used to test what each group is worth.
GROUPS: dict[str, tuple[str, ...]] = {
    "lineup": ("d_lu_woba", "d_lu_k", "d_lu_bb", "d_lu_iso", "d_lu_exp", "d_lu_vs_usual"),
    "left_right": ("d_lu_hand", "d_sp_vs_lu", "d_mu_woba", "d_mu_k"),
    "history": ("d_hist", "d_hist_pa"),
    "starter_stuff": ("d_sp_woba", "d_sp_whiff", "d_sp_csw", "d_sp_gb", "d_sp_speed", "d_sp_speed_trend"),
    "bullpen": ("d_bp_woba", "d_bp_k", "d_bp_tired"),
    "defense": ("d_def_der",),
    "statcast": ("d_lu_hard", "d_sp_hard", "d_lu_xgap", "d_sp_xgap", "d_lu_oaa", "d_lu_speed"),
    "injuries": ("d_il_bat", "d_il_pit", "d_il_n"),
    "recent_form": ("d_lu_hot", "d_sp_hot"),
    "age": ("d_lu_age", "d_sp_age"),
    "setting": ("park_rf", "temp", "wind_out", "ump_runs"),
    "expected_runs": ("d_xr", "x_total"),
}


# ---------------------------------------------------------------------------
# Totals "as of" a game
# ---------------------------------------------------------------------------


def asof_totals(events: pd.DataFrame, queries: pd.DataFrame, by, sums, *, career: bool = False) -> pd.DataFrame:
    """Totals of ``sums`` from games strictly before each queried game.

    ``events`` holds counted columns per game. ``queries`` asks "what had this
    player (or pair, or team) done before this game?". Both carry ``season``
    and ``order`` plus the columns named in ``by``.

    By default this season's games count in full and earlier seasons count
    for less the older they are. With ``career`` every earlier game counts
    the same.
    """
    by, sums = list(by), list(sums)
    keys = by + ["season", "order"]
    ev = (events.groupby(keys, as_index=False, sort=False)[sums].sum()
          .sort_values("order", kind="stable").reset_index(drop=True))
    q = queries[keys].reset_index(drop=True)
    q["_row"] = np.arange(len(q))
    group = by if career else by + ["season"]
    running = ev.groupby(group, sort=False)[sums].cumsum()
    for column in group + ["order"]:
        running[column] = ev[column].to_numpy()
    got = pd.merge_asof(q.sort_values("order", kind="stable"), running, on="order", by=group,
                        allow_exact_matches=False)
    total = got.sort_values("_row")[sums].fillna(0.0).to_numpy()
    if not career:
        season_tot = ev.groupby(by + ["season"], as_index=False, sort=False)[sums].sum()
        pairs = q[by + ["season"]].drop_duplicates()
        earlier = pairs.merge(season_tot.rename(columns={"season": "season_past"}), on=by)
        earlier = earlier[earlier["season_past"] < earlier["season"]]
        weights = _season_weight((earlier["season"] - earlier["season_past"]).to_numpy())
        weighted = earlier[sums].mul(weights, axis=0)
        for column in by + ["season"]:
            weighted[column] = earlier[column].to_numpy()
        past = weighted.groupby(by + ["season"], as_index=False, sort=False).sum()
        past = q[by + ["season", "_row"]].merge(past, on=by + ["season"], how="left").sort_values("_row")
        total = total + past[sums].fillna(0.0).to_numpy()
    return pd.DataFrame(total, columns=sums, index=queries.index)


def window_totals(events: pd.DataFrame, queries: pd.DataFrame, by, sums, days: int) -> pd.DataFrame:
    """Totals of ``sums`` over the ``days`` days before each queried game's date.

    The day of the game itself is never included. Both frames carry ``day``
    (a running day number) plus the columns named in ``by``.
    """
    by, sums = list(by), list(sums)
    daily = (events.groupby(by + ["day"], as_index=False, sort=False)[sums].sum()
             .sort_values("day", kind="stable").reset_index(drop=True))
    running = daily.groupby(by, sort=False)[sums].cumsum()
    for column in by + ["day"]:
        running[column] = daily[column].to_numpy()
    q = queries[by + ["day"]].reset_index(drop=True)
    q["_row"] = np.arange(len(q))
    total = np.zeros((len(q), len(sums)))
    for back, sign in ((1, 1.0), (days + 1, -1.0)):
        probe = q.assign(day=q["day"] - back).sort_values("day", kind="stable")
        got = pd.merge_asof(probe, running, on="day", by=by).sort_values("_row")
        total += sign * got[sums].fillna(0.0).to_numpy()
    return pd.DataFrame(total, columns=sums, index=queries.index)


def _shrunk(num, den, prior, k):
    """A rate pulled toward ``prior`` with the weight of ``k`` observations."""
    return (num + k * prior) / (den + k)


def _prior_season(table: pd.DataFrame, extra: list[int] | None = None) -> pd.DataFrame:
    """Give each season the row of the season before it.

    The first season in the data has no "season before". It only warms up the
    running numbers, so its own row stands in. Seasons listed in ``extra``
    that are later than the data take the latest row available.
    """
    table = table.sort_values("season")
    first = table[table["season"] == table["season"].min()]
    shifted = table.assign(season=table["season"] + 1)
    out = pd.concat([first, shifted], ignore_index=True)
    last = table[table["season"] == table["season"].max()]
    for season in sorted(set(extra or []) - set(out["season"])):
        out = pd.concat([out, last.assign(season=season)], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


class _Ids:
    """Player ids as small integers, so joins are fast and of one type on every pandas version."""

    def __init__(self, *columns: pd.Series):
        values = pd.concat([c.dropna().astype(str) for c in columns], ignore_index=True).unique()
        self.lookup = {value: code for code, value in enumerate(values)}

    def __call__(self, column: pd.Series) -> pd.Series:
        text = column.astype(object).where(column.notna(), None)
        return text.map(lambda v: self.lookup.get(str(v), -1) if v is not None else -1).astype("int64")


def _ages(player: pd.Series, day: pd.Series, people: pd.DataFrame) -> np.ndarray:
    """Age in years on the day of the game. A player with no birth date gets a typical age."""
    if not len(people) or "birth_date" not in people.columns:
        return np.full(len(player), TYPICAL_AGE)
    born = pd.to_datetime(people["birth_date"], errors="coerce")
    born_day = pd.Series((born - pd.Timestamp("2000-01-01")).dt.days.to_numpy(), index=people["pid"].to_numpy())
    born_day = born_day[~born_day.index.duplicated()]
    age = (day.to_numpy(dtype=float) - player.map(born_day).to_numpy(dtype=float)) / 365.25
    return np.where(np.isfinite(age) & (age > 15) & (age < 55), age, TYPICAL_AGE)


def _majority(frame: pd.DataFrame, key: str, value: str, weight: str) -> pd.DataFrame:
    counts = frame.groupby([key, value], sort=False)[weight].sum().reset_index()
    counts = counts[counts[value].isin(["L", "R"])]
    wide = counts.pivot(index=key, columns=value, values=weight).reindex(columns=["L", "R"]).fillna(0.0)
    return wide


def _handedness(pa: pd.DataFrame, people: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Batting side (L, R, or S for switch) per hitter and throwing hand per pitcher.

    Taken from how each player actually stood in his plate appearances, with
    the roster listing as the fallback for a player who has none yet.
    """
    sides = _majority(pa, "batter", "bat_side", "pa")
    total = sides.sum(axis=1).replace(0.0, np.nan)
    minority = sides.min(axis=1) / total
    bats = pd.Series(np.where(sides["L"] >= sides["R"], "L", "R"), index=sides.index)
    bats[(minority >= 0.10) & (sides.min(axis=1) >= 10)] = "S"
    hands = _majority(pa, "pitcher", "pit_hand", "pa")
    throws = pd.Series(np.where(hands["L"] > hands["R"], "L", "R"), index=hands.index)
    if len(people):
        listed = people.set_index("pid")
        known_bats = listed["bats"][listed["bats"].isin(["L", "R", "S"])]
        known_throws = listed["throws"][listed["throws"].isin(["L", "R"])]
        bats = pd.concat([bats, known_bats[~known_bats.index.isin(bats.index)]])
        throws = pd.concat([throws, known_throws[~known_throws.index.isin(throws.index)]])
    return bats, throws


def _league(pa: pd.DataFrame, games: pd.DataFrame, seasons: list[int]) -> dict[str, pd.DataFrame]:
    """League averages by season, each one taken from the season before."""
    tot = pa.groupby("season")[["woba_num", "woba_den", "pa", "ab", "so", "bb", "h2", "h3", "hr", "pitches",
                                "whiffs", "called", "tracked", "gb", "ev_n", "hard", "fb_speed_sum", "fb_n",
                                "bip", "bip_hit"]].sum()
    hard = (tot["hard"] / tot["ev_n"].replace(0.0, np.nan))
    base = pd.DataFrame({
        "lg_woba": tot["woba_num"] / tot["woba_den"],
        "lg_k": tot["so"] / tot["pa"],
        "lg_bb": tot["bb"] / tot["pa"],
        "lg_iso": (tot["h2"] + 2 * tot["h3"] + 3 * tot["hr"]) / tot["ab"],
        "lg_whiff": tot["whiffs"] / tot["pitches"],
        "lg_csw": (tot["whiffs"] + tot["called"]) / tot["pitches"],
        "lg_gb": tot["gb"] / tot["tracked"].replace(0.0, np.nan),
        "lg_hard": hard.fillna(hard.mean() if hard.notna().any() else 0.38),
        "lg_speed": tot["fb_speed_sum"] / tot["fb_n"].replace(0.0, np.nan),
        "lg_bip_hit": tot["bip_hit"] / tot["bip"],
    }).reset_index()
    pitchers_batting = pa[pa["batter_is_pitcher"]].groupby("season")[["woba_num", "woba_den"]].sum()
    base["lg_woba_p"] = base["season"].map(
        _shrunk(pitchers_batting["woba_num"], pitchers_batting["woba_den"], PITCHER_HITTING, 300.0)
    ).fillna(PITCHER_HITTING)
    final = games[(games["status"] == "final") & games["home_score"].notna()]
    runs = (final["home_score"] + final["away_score"]).groupby(final["season"]).agg(["sum", "size"])
    base["lg_runs"] = base["season"].map(runs["sum"] / (2.0 * runs["size"])).fillna(4.5)
    out = {"base": _prior_season(base, seasons)}

    def cell_table(keys: list[str], name: str) -> pd.DataFrame:
        cells = pa.groupby(["season", *keys])[["woba_num", "woba_den"]].sum().reset_index()
        cells[name] = cells["woba_num"] / cells["woba_den"].replace(0.0, np.nan)
        return cells[["season", *keys, name]]

    # Actual batting side against actual throwing hand: the baseline for one matchup.
    cell = cell_table(["bat_side", "pit_hand"], "lg_cell")
    cell = cell[cell["bat_side"].isin(["L", "R"]) & cell["pit_hand"].isin(["L", "R"])]
    out["cell"] = pd.concat([_prior_season(part, seasons) for _, part in cell.groupby(["bat_side", "pit_hand"])],
                            ignore_index=True)
    # How much a left-, right- or switch-hitter gains or loses against each hand.
    by_class = cell_table(["bats", "pit_hand"], "w")
    by_class = by_class[by_class["bats"].isin(["L", "R", "S"]) & by_class["pit_hand"].isin(["L", "R"])]
    overall = cell_table(["bats"], "w_all")
    delta = by_class.merge(overall, on=["season", "bats"])
    delta["delta_b"] = delta["w"] - delta["w_all"]
    out["delta_b"] = pd.concat(
        [_prior_season(part[["season", "bats", "pit_hand", "delta_b"]], seasons)
         for _, part in delta.groupby(["bats", "pit_hand"])], ignore_index=True)
    # The same from the pitcher's side: left- and right-handed pitchers against each batting side.
    overall_p = cell_table(["pit_hand"], "w_all")
    delta_p = cell.merge(overall_p, on=["season", "pit_hand"])
    delta_p["delta_p"] = delta_p["lg_cell"] - delta_p["w_all"]
    out["delta_p"] = pd.concat(
        [_prior_season(part[["season", "bat_side", "pit_hand", "delta_p"]], seasons)
         for _, part in delta_p.groupby(["bat_side", "pit_hand"])], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------


def park_hitting_factors(pa: pd.DataFrame, games: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """How much each ballpark inflates or deflates hitting, by season, from the three seasons before.

    For each park: hitting (wOBA) in games played there, divided by hitting
    in the same home club's road games, with both teams' hitters counted in
    each. A park with little history is pulled toward 1.0 (no effect).
    Returns ``venue``, ``season`` and ``woba_pf``.
    """
    where = games[["game_id", "season", "home", "away"]].merge(extras[["game_id", "venue"]], on="game_id", how="left")
    per_game = pa.groupby("game_id")[["woba_num", "woba_den"]].sum().reset_index().merge(where, on="game_id")
    per_game = per_game[per_game["venue"].notna()]
    at_park = per_game.groupby(["venue", "season", "home"], as_index=False).agg(
        num=("woba_num", "sum"), den=("woba_den", "sum"), n=("game_id", "size"))
    road = per_game.groupby(["season", "away"], as_index=False).agg(road_num=("woba_num", "sum"), road_den=("woba_den", "sum"))
    at_park = at_park.merge(road.rename(columns={"away": "home"}), on=["season", "home"], how="left")
    at_park["expected"] = at_park["den"] * at_park["road_num"] / at_park["road_den"].replace(0.0, np.nan)
    rows = []
    for season in sorted(int(s) for s in games["season"].unique()):
        window = at_park[(at_park["season"] < season) & (at_park["season"] >= season - 3)]
        if window.empty:
            continue
        agg = window.groupby("venue")[["num", "expected", "n"]].sum()
        ratio = (agg["num"] / agg["expected"].replace(0.0, np.nan)).fillna(1.0)
        factor = (ratio * agg["n"] + K_PARK) / (agg["n"] + K_PARK)
        rows.append(pd.DataFrame({"venue": factor.index, "season": season, "woba_pf": factor.to_numpy()}))
    if not rows:
        return pd.DataFrame(columns=["venue", "season", "woba_pf"])
    return pd.concat(rows, ignore_index=True)


def _park_neutral(pa: pd.DataFrame, games: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """Scale each plate appearance's hitting value by the park it happened in.

    A hitter who plays half his games in a park that inflates hitting would
    otherwise look better than he is everywhere else, and the pitchers who
    work there would look worse. The park's effect is always measured on
    earlier seasons, so nothing from the season in question is used.
    """
    factors = park_hitting_factors(pa, games, extras)
    if factors.empty:
        return pa
    venue = pa[["game_id", "season"]].merge(extras[["game_id", "venue"]].drop_duplicates("game_id"), on="game_id", how="left")
    factor = venue.merge(factors, on=["venue", "season"], how="left")["woba_pf"].fillna(1.0).to_numpy()
    pa = pa.copy()
    pa["woba_num"] = pa["woba_num"] / factor
    return pa


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: each team, its opponent and the pitcher it faces."""
    base = ["game_id", "season", "order", "day", "status"]
    home = games[base + ["home", "away", "away_sp", "home_sp"]].rename(
        columns={"home": "team", "away": "opp", "away_sp": "opp_sp", "home_sp": "own_sp"}).assign(is_home=True)
    away = games[base + ["away", "home", "home_sp", "away_sp"]].rename(
        columns={"away": "team", "home": "opp", "home_sp": "opp_sp", "away_sp": "own_sp"}).assign(is_home=False)
    return pd.concat([home, away], ignore_index=True)


def choose_lineups(team_games: pd.DataFrame, lineups: pd.DataFrame, rosters: pd.DataFrame,
                   mode: str = "posted") -> pd.DataFrame:
    """The nine hitters to use for every team-game, with a flag for "posted".

    A team-game without a posted lineup (or every team-game in ``projected``
    mode) gets a guessed one. The starting point is the lineup from the
    team's most recent game this season against a starter of the same
    throwing hand: a real lineup, days off included. Anyone in it who is no
    longer on the roster is replaced by whoever started most often over the
    last few such games. At the start of a season there is no recent game, so
    the nine who started most often at the end of last season are used.
    """
    if mode not in ("posted", "projected"):
        raise ValueError("lineup mode must be 'posted' or 'projected'")
    tg = team_games.sort_values(["team", "order"]).reset_index(drop=True)
    have = lineups.groupby(["game_id", "team"]).size().rename("n").reset_index()
    tg = tg.merge(have, on=["game_id", "team"], how="left")
    tg["has"] = tg["n"].fillna(0) >= 9
    use_posted = tg["has"] if mode == "posted" else pd.Series(False, index=tg.index)
    posted = lineups.merge(tg.loc[use_posted, ["game_id", "team"]], on=["game_id", "team"])
    posted = posted[["game_id", "team", "slot", "batter", "position"]].assign(posted=1.0)

    # Number each team's games that have a lineup, separately for left- and
    # right-handed opposing starters, so that "the game before last against a
    # lefty" can be looked up directly.
    group = [tg["team"], tg["opp_hand"]]
    tg["seen"] = tg["has"].astype(int).groupby(group).cumsum() - tg["has"].astype(int)
    sources = tg.loc[tg["has"], ["team", "opp_hand", "seen", "game_id", "season"]].rename(
        columns={"seen": "seq", "game_id": "source", "season": "source_season"})
    need = tg.loc[~use_posted, ["game_id", "team", "opp_hand", "seen", "season"]]
    parts = []
    for back in range(1, RECENT_LINEUPS + 1):
        step = need.assign(seq=need["seen"] - back, weight=RECENCY ** (back - 1), latest=(back == 1))
        parts.append(step[step["seq"] >= 0])
    if not parts or not len(need):
        return posted
    wanted = pd.concat(parts, ignore_index=True).merge(sources, on=["team", "opp_hand", "seq"])
    # The latest lineup from this season outranks everything earlier put together.
    first_choice = wanted["latest"] & (wanted["source_season"] == wanted["season"])
    wanted["weight"] = np.where(first_choice, 100.0, wanted["weight"])
    pool = wanted[["game_id", "team", "source", "weight"]].merge(
        lineups.rename(columns={"game_id": "source"}), on=["source", "team"])
    if len(pool) and len(rosters):
        active = rosters[["game_id", "team", "pid"]].drop_duplicates().assign(on_roster=True)
        has_roster = rosters[["game_id", "team"]].drop_duplicates().assign(roster_known=True)
        pool = pool.merge(active, left_on=["game_id", "team", "batter"], right_on=["game_id", "team", "pid"], how="left")
        pool = pool.merge(has_roster, on=["game_id", "team"], how="left")
        on_roster = pool["on_roster"].astype("boolean").fillna(False).astype(bool)
        roster_known = pool["roster_known"].astype("boolean").fillna(False).astype(bool)
        pool = pool[on_roster | ~roster_known]
    if pool.empty:
        return posted
    pool = pool.assign(slot_weight=pool["slot"] * pool["weight"])
    pool = pool.sort_values("weight", ascending=False, kind="stable")  # most recent game first
    picked = pool.groupby(["game_id", "team", "batter"], as_index=False, sort=False).agg(
        weight=("weight", "sum"), slot_weight=("slot_weight", "sum"), position=("position", "first"))
    picked["usual_slot"] = picked["slot_weight"] / picked["weight"]
    picked = picked.sort_values(["game_id", "team", "weight", "usual_slot"], ascending=[True, True, False, True], kind="stable")
    picked = picked[picked.groupby(["game_id", "team"]).cumcount() < 9]
    picked = picked.sort_values(["game_id", "team", "usual_slot"], kind="stable")
    picked["slot"] = picked.groupby(["game_id", "team"]).cumcount() + 1
    guessed = picked[["game_id", "team", "slot", "batter", "position"]].assign(posted=0.0)
    return pd.concat([posted, guessed], ignore_index=True)


# ---------------------------------------------------------------------------
# The main build
# ---------------------------------------------------------------------------

_BAT_SUMS = ["woba_num", "woba_den", "pa", "ab", "so", "bb", "h2", "h3", "hr", "ev_n", "hard"]
_PIT_SUMS = ["woba_num", "woba_den", "pa", "so", "bb", "pitches", "whiffs", "called", "tracked", "gb",
             "ev_n", "hard", "fb_speed_sum", "fb_n"]


def build_player_features(games: pd.DataFrame, tables: dict[str, pd.DataFrame], *,
                          lineup_mode: str = "posted") -> pd.DataFrame:
    """One row per game with every player-level input.

    ``tables`` needs ``lineups``, ``rosters``, ``plate_appearances``,
    ``game_extras`` and ``people``. ``il_spells`` (injured list stretches) and
    the three Statcast season tables are used when present. Works for
    finished and scheduled games alike, through the same code.
    """
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    games["order"] = order_key(games["date"], games["number"])
    games["day"] = (games["date"] - pd.Timestamp("2000-01-01")).dt.days.astype("int64")
    games["season"] = games["season"].astype("int64")
    games = games.sort_values("order").reset_index(drop=True)
    seasons = sorted(int(s) for s in games["season"].unique())

    ids = _Ids(tables["plate_appearances"]["batter"], tables["plate_appearances"]["pitcher"],
               tables["lineups"]["player_id"], tables["rosters"]["player_id"],
               games["home_sp_id"], games["away_sp_id"],
               tables["people"]["player_id"] if len(tables["people"]) else pd.Series([], dtype=object))
    games["home_sp"], games["away_sp"] = ids(games["home_sp_id"]), ids(games["away_sp_id"])
    when = games[["game_id", "season", "order", "day"]]

    people = tables["people"].copy()
    if len(people):
        people["pid"] = ids(people["player_id"])
    pa = tables["plate_appearances"].merge(when, on="game_id", how="inner")
    pa["batter"], pa["pitcher"] = ids(pa["batter"]), ids(pa["pitcher"])
    pa = _park_neutral(pa, games, tables["game_extras"])
    bats, throws = _handedness(pa, people)
    if len(people):
        listed_pitchers = set(people.loc[people["position"] == "P", "pid"])
    else:
        listed_pitchers = set()
    pa["bats"] = pa["batter"].map(bats)
    pa["batter_is_pitcher"] = pa["batter"].isin(listed_pitchers)
    league = _league(pa, games, seasons)

    lineups = tables["lineups"].rename(columns={"player_id": "batter"}).copy()
    lineups["batter"] = ids(lineups["batter"])
    rosters = tables["rosters"].copy()
    rosters["pid"] = ids(rosters["player_id"])

    tg = _team_games(games)
    tg["opp_hand"] = tg["opp_sp"].map(throws).fillna("R")
    tg["own_hand"] = tg["own_sp"].map(throws).fillna("R")

    # ---- starting pitchers -------------------------------------------------
    sp = tg[["game_id", "team", "season", "order", "day", "own_sp", "own_hand"]].rename(columns={"own_sp": "pitcher"})
    p_all = asof_totals(pa, sp, ["pitcher"], _PIT_SUMS)
    sp = sp.merge(league["base"], on="season", how="left")
    sp["p_woba"] = _shrunk(p_all["woba_num"], p_all["woba_den"], sp["lg_woba"], K_PIT).to_numpy()
    sp["p_k"] = _shrunk(p_all["so"], p_all["pa"], sp["lg_k"], K_PIT_K).to_numpy()
    for side in ("L", "R"):
        split = asof_totals(pa, sp.assign(bat_side=side), ["pitcher", "bat_side"], ["woba_num", "woba_den"])
        shift = sp[["season", "own_hand"]].rename(columns={"own_hand": "pit_hand"}).assign(bat_side=side).merge(
            league["delta_p"], on=["season", "bat_side", "pit_hand"], how="left")["delta_p"].fillna(0.0).to_numpy()
        sp[f"p_woba_{side}"] = _shrunk(split["woba_num"], split["woba_den"], sp["p_woba"] + shift, K_PIT_SPLIT).to_numpy()
    sp["sp_woba"] = sp["p_woba"]
    sp["sp_whiff"] = _shrunk(p_all["whiffs"], p_all["pitches"], sp["lg_whiff"], K_WHIFF).to_numpy()
    sp["sp_csw"] = _shrunk(p_all["whiffs"] + p_all["called"], p_all["pitches"], sp["lg_csw"], K_WHIFF).to_numpy()
    sp["sp_gb"] = _shrunk(p_all["gb"], p_all["tracked"], sp["lg_gb"].fillna(0.44), K_GB).to_numpy()
    sp["sp_hard"] = _shrunk(p_all["hard"], p_all["ev_n"], sp["lg_hard"], K_HARD).to_numpy()
    lg_speed = sp["lg_speed"].fillna(92.0)
    # Fastball speed is compared with the league so that a league-wide rise over the years is not read as skill.
    sp["sp_speed"] = (_shrunk(p_all["fb_speed_sum"], p_all["fb_n"], lg_speed, K_SPEED) - lg_speed).clip(-8.0, 8.0).to_numpy()

    # Appearances, one row per pitcher per game: used for recent speed, workload and roles.
    apps = pa.groupby(["pitcher", "pit_team", "game_id", "season", "order", "day"], as_index=False, sort=False).agg(
        bf=("pa", "sum"), pitches=("pitches", "sum"), fb_speed_sum=("fb_speed_sum", "sum"), fb_n=("fb_n", "sum"),
        started=("vs_starter", "max"), so=("so", "sum"), woba_num=("woba_num", "sum"), woba_den=("woba_den", "sum"))
    apps["started"] = apps["started"].astype(float)
    apps = apps.sort_values(["pitcher", "order"]).reset_index(drop=True)
    by_pitcher = apps.groupby("pitcher", sort=False)
    apps["fb3_sum"] = by_pitcher["fb_speed_sum"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    apps["fb3_n"] = by_pitcher["fb_n"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    apps["w3_num"] = by_pitcher["woba_num"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    apps["w3_den"] = by_pitcher["woba_den"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    apps["start_share"] = by_pitcher["started"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    apps["start_bf"] = apps["bf"] * apps["started"]
    recent = pd.merge_asof(
        sp[["pitcher", "order"]].assign(_row=np.arange(len(sp))).sort_values("order", kind="stable"),
        apps[["pitcher", "order", "fb3_sum", "fb3_n", "w3_num", "w3_den"]].sort_values("order", kind="stable"),
        on="order", by="pitcher", allow_exact_matches=False).sort_values("_row")
    recent_speed = recent["fb3_sum"] / recent["fb3_n"].replace(0.0, np.nan)
    season_speed = p_all["fb_speed_sum"] / p_all["fb_n"].replace(0.0, np.nan)
    enough = (recent["fb3_n"].to_numpy() >= 20) & (p_all["fb_n"].to_numpy() >= 100)
    sp["sp_speed_trend"] = np.where(enough, recent_speed.to_numpy() - season_speed.to_numpy(), 0.0)
    sp["sp_speed_trend"] = sp["sp_speed_trend"].fillna(0.0).clip(-3.0, 3.0)
    # His last three outings against his own longer record: above zero means he has been hit harder lately.
    sp["sp_hot"] = (_shrunk(recent["w3_num"].fillna(0.0).to_numpy(), recent["w3_den"].fillna(0.0).to_numpy(),
                            sp["p_woba"].to_numpy(), K_HOT) - sp["p_woba"].to_numpy())
    sp["sp_age"] = _ages(sp["pitcher"], sp["day"], people)
    workload = asof_totals(apps, sp, ["pitcher"], ["start_bf", "started"])
    lg_bf_start = 0.62 * BF_PER_GAME
    sp["sp_share"] = (_shrunk(workload["start_bf"], workload["started"], lg_bf_start, K_STARTS)
                      / BF_PER_GAME).clip(0.35, 0.80).to_numpy()
    statcast = _statcast_prior(tables, ids)
    sp = sp.merge(statcast["pit"].rename(columns={"pid": "pitcher", "xgap": "sp_xgap"}), on=["pitcher", "season"], how="left")
    sp["sp_xgap"] = pd.to_numeric(sp["sp_xgap"], errors="coerce").fillna(0.0)

    # ---- hitters in the lineup -------------------------------------------
    lu = choose_lineups(tg, lineups, rosters, lineup_mode)
    lu = lu.merge(tg[["game_id", "team", "season", "order", "day", "opp", "opp_sp", "opp_hand"]], on=["game_id", "team"])
    lu = lu.merge(league["base"], on="season", how="left").reset_index(drop=True)
    b_all = asof_totals(pa, lu, ["batter"], _BAT_SUMS)
    b_hand = asof_totals(pa, lu.assign(pit_hand=lu["opp_hand"]), ["batter", "pit_hand"], ["woba_num", "woba_den"])
    is_pitcher = (lu["position"] == "P") | lu["batter"].isin(listed_pitchers)
    prior = np.where(is_pitcher, lu["lg_woba_p"], UNKNOWN_HITTER * lu["lg_woba"])
    lu["b_all"] = _shrunk(b_all["woba_num"], b_all["woba_den"], prior, K_BAT).to_numpy()
    lu["bats"] = lu["batter"].map(bats).fillna("R")
    other_hand = np.where(lu["opp_hand"] == "L", "R", "L")

    def platoon_shift(hand) -> np.ndarray:
        frame = pd.DataFrame({"season": lu["season"], "bats": lu["bats"], "pit_hand": hand})
        return frame.merge(league["delta_b"], on=["season", "bats", "pit_hand"], how="left")["delta_b"].fillna(0.0).to_numpy()

    lu["b_hand"] = _shrunk(b_hand["woba_num"], b_hand["woba_den"], lu["b_all"] + platoon_shift(lu["opp_hand"]),
                           K_BAT_SPLIT).to_numpy()
    lu["b_other"] = _shrunk(b_all["woba_num"] - b_hand["woba_num"], b_all["woba_den"] - b_hand["woba_den"],
                            lu["b_all"] + platoon_shift(other_hand), K_BAT_SPLIT).to_numpy()
    lu["b_k"] = _shrunk(b_all["so"], b_all["pa"], lu["lg_k"], K_BAT_K).to_numpy()
    lu["b_bb"] = _shrunk(b_all["bb"], b_all["pa"], lu["lg_bb"], K_BAT_BB).to_numpy()
    lu["b_iso"] = _shrunk(b_all["h2"] + 2 * b_all["h3"] + 3 * b_all["hr"], b_all["ab"], lu["lg_iso"], K_BAT_ISO).to_numpy()
    lu["b_exp"] = np.log1p(b_all["pa"]).to_numpy()
    lately = window_totals(pa, lu, ["batter"], ["woba_num", "woba_den"], HOT_DAYS)
    lu["b_hot"] = (_shrunk(lately["woba_num"], lately["woba_den"], lu["b_all"], K_HOT) - lu["b_all"]).to_numpy()
    lu["b_age"] = _ages(lu["batter"], lu["day"], people)
    lu["b_hard"] = _shrunk(b_all["hard"], b_all["ev_n"], lu["lg_hard"], K_BAT_HARD).to_numpy()
    lu = lu.merge(statcast["bat"].rename(columns={"pid": "batter", "xgap": "b_xgap"}), on=["batter", "season"], how="left")
    lu = lu.merge(statcast["field"].rename(columns={"pid": "batter", "oaa": "b_oaa"}), on=["batter", "season"], how="left")
    lu = lu.merge(statcast["speed"].rename(columns={"pid": "batter", "sprint_speed": "b_speed"}), on=["batter", "season"], how="left")
    lu["b_xgap"] = pd.to_numeric(lu["b_xgap"], errors="coerce").fillna(0.0)
    lu["b_oaa"] = pd.to_numeric(lu["b_oaa"], errors="coerce").fillna(0.0)
    lu["b_speed"] = pd.to_numeric(lu["b_speed"], errors="coerce").fillna(27.0)
    # The side he will stand on against this starter: a switch hitter takes the opposite side.
    lu["side"] = np.where(lu["bats"] == "S", np.where(lu["opp_hand"] == "L", "R", "L"), lu["bats"])

    # ---- hitter against starter ------------------------------------------
    opp = sp[["game_id", "team", "p_woba_L", "p_woba_R", "p_k"]].rename(columns={"team": "opp"})
    lu = lu.merge(opp, on=["game_id", "opp"], how="left")
    lu["p_side"] = np.where(lu["side"] == "L", lu["p_woba_L"], lu["p_woba_R"])
    cell = lu[["season", "side", "opp_hand"]].rename(columns={"side": "bat_side", "opp_hand": "pit_hand"}).merge(
        league["cell"], on=["season", "bat_side", "pit_hand"], how="left")["lg_cell"].to_numpy()
    lu["lg_cell"] = np.where(np.isnan(cell), lu["lg_woba"], cell)
    lu["m"] = lu["b_hand"] * lu["p_side"] / lu["lg_cell"]
    lu["m_k"] = lu["b_k"] * lu["p_k"] / lu["lg_k"]
    history = asof_totals(pa, lu.assign(pitcher=lu["opp_sp"]), ["batter", "pitcher"],
                          ["woba_num", "woba_den"], career=True)
    met = (lu["opp_sp"] >= 0).to_numpy()
    lu["hist_pa"] = np.where(met, history["woba_den"], 0.0)
    lu["m_hist"] = np.where(met, _shrunk(history["woba_num"], history["woba_den"], lu["m"], K_HISTORY), lu["m"])
    lu["hist"] = lu["m_hist"] - lu["m"]

    weight = SLOT_WEIGHTS[np.clip(lu["slot"].to_numpy(dtype=int), 1, 9) - 1]
    lu["w"] = weight / pd.Series(weight).groupby([lu["game_id"], lu["team"]]).transform("sum").to_numpy()

    def lineup_average(column: str) -> pd.Series:
        return (lu[column] * lu["w"]).groupby([lu["game_id"], lu["team"]]).sum()

    team = pd.DataFrame({
        "lu_woba": lineup_average("b_all"), "lu_hand": lineup_average("b_hand"), "lu_other": lineup_average("b_other"),
        "lu_k": lineup_average("b_k"), "lu_bb": lineup_average("b_bb"), "lu_iso": lineup_average("b_iso"),
        "lu_exp": lineup_average("b_exp"), "mu_woba": lineup_average("m"), "mu_k": lineup_average("m_k"),
        "lu_hard": lineup_average("b_hard"), "lu_xgap": lineup_average("b_xgap"),
        "lu_hot": lineup_average("b_hot"), "lu_age": lineup_average("b_age"),
        "lu_speed": lineup_average("b_speed"), "lu_oaa": lu.groupby(["game_id", "team"])["b_oaa"].sum(),
        "mu_hist": lineup_average("m_hist"), "hist": lineup_average("hist"),
        "hist_pa": np.log1p(lu.groupby(["game_id", "team"])["hist_pa"].sum()),
        "opp_sp_vs_lu": lineup_average("p_side"), "lu_posted": lu.groupby(["game_id", "team"])["posted"].min(),
        "lu_n": lu.groupby(["game_id", "team"]).size().astype(float),
    }).reset_index()
    team = tg.merge(team, on=["game_id", "team"], how="left")
    team["lu_posted"] = team["lu_posted"].fillna(0.0)
    team = team.sort_values(["team", "order"]).reset_index(drop=True)
    usual = team.groupby(["team", "season"])["lu_woba"].transform(
        lambda s: s.shift(1).rolling(USUAL_WINDOW, min_periods=5).mean())
    team["lu_vs_usual"] = (team["lu_woba"] - usual).fillna(0.0)

    sp_cols = ["sp_woba", "sp_whiff", "sp_csw", "sp_gb", "sp_hard", "sp_speed", "sp_speed_trend", "sp_share",
               "sp_xgap", "sp_hot", "sp_age"]
    team = team.merge(sp[["game_id", "team", *sp_cols]], on=["game_id", "team"], how="left")
    # How this team's starter fares against the sides of the lineup he faces.
    facing = team[["game_id", "team", "opp_sp_vs_lu"]].rename(columns={"team": "opp", "opp_sp_vs_lu": "sp_vs_lu"})
    team = team.merge(facing, on=["game_id", "opp"], how="left")

    # ---- bullpen, defense, setting ---------------------------------------
    team = team.merge(_bullpens(tg, rosters, apps, pa, league, throws), on=["game_id", "team"], how="left")
    team = team.merge(_defense(tg, pa, league), on=["game_id", "team"], how="left")
    team = team.merge(_injured(tg, tables.get("il_spells"), pa, league, ids), on=["game_id", "team"], how="left")
    for column in ("il_bat", "il_pit", "il_n"):
        team[column] = pd.to_numeric(team[column], errors="coerce").fillna(0.0)
    setting = _setting(games, tables["game_extras"], league)

    # ---- expected runs -----------------------------------------------------
    team = team.merge(league["base"][["season", "lg_woba", "lg_runs"]], on="season", how="left")
    other = team[["game_id", "team", "sp_share", "bp_woba", "bp_left"]].rename(
        columns={"team": "opp", "sp_share": "opp_sp_share", "bp_woba": "opp_bp_woba", "bp_left": "opp_bp_left"})
    team = team.merge(other, on=["game_id", "opp"], how="left")
    vs_left = np.where(team["opp_hand"] == "L", team["lu_hand"], team["lu_other"])
    vs_right = np.where(team["opp_hand"] == "L", team["lu_other"], team["lu_hand"])
    left_share = team["opp_bp_left"].fillna(0.28)
    # With no bullpen information at all (a game far in the future), an average bullpen stands in.
    pen_quality = team["opp_bp_woba"].fillna(team["lg_woba"])
    against_pen = (left_share * vs_left + (1.0 - left_share) * vs_right) * pen_quality / team["lg_woba"]
    share = team["opp_sp_share"].fillna(0.6)
    expected_woba = share * team["mu_hist"] + (1.0 - share) * against_pen
    team = team.merge(setting[["game_id", "park_rf"]], on="game_id", how="left")
    team["xr"] = ((team["lg_runs"] + PA_PER_GAME * (expected_woba - team["lg_woba"]) / WOBA_SCALE)
                  * team["park_rf"].fillna(1.0)).clip(lower=1.5)

    # ---- one row per game --------------------------------------------------
    out = games[["game_id"]].copy()
    stats = [s for s in TEAM_PLAYER_STATS] + ["lu_posted", "lu_n"]
    for side, flag in (("home", True), ("away", False)):
        part = team.loc[team["is_home"] == flag, ["game_id", *stats]]
        out = out.merge(part.rename(columns={s: f"{side}_{s}" for s in stats}), on="game_id", how="left")
    for stat in TEAM_PLAYER_STATS:
        out[f"d_{stat}"] = out[f"home_{stat}"] - out[f"away_{stat}"]
    out["x_total"] = out["home_xr"] + out["away_xr"]
    out = out.merge(setting, on="game_id", how="left")
    return out


# ---------------------------------------------------------------------------
# Bullpen
# ---------------------------------------------------------------------------


def _bullpens(tg: pd.DataFrame, rosters: pd.DataFrame, apps: pd.DataFrame, pa: pd.DataFrame,
              league: dict, throws: pd.Series) -> pd.DataFrame:
    """Quality and tiredness of each team's available relievers before each game."""
    staff = rosters[rosters["role"].isin(["RP", "BP", "SP"])][["game_id", "team", "pid"]].rename(columns={"pid": "pitcher"})
    cand = staff.merge(tg[["game_id", "team", "season", "order", "day", "own_sp"]], on=["game_id", "team"])
    # Team-games with no roster listing fall back to whoever pitched in relief for the club in the last 20 days.
    listed = set(zip(cand["game_id"], cand["team"]))
    missing = tg[[key not in listed for key in zip(tg["game_id"], tg["team"])]]
    if len(missing):
        relief = apps.loc[apps["started"] == 0, ["pitcher", "pit_team", "day"]].rename(
            columns={"pit_team": "team", "day": "app_day"})
        recent = missing[["game_id", "team", "season", "order", "day", "own_sp"]].merge(relief, on="team")
        recent = recent[(recent["day"] - recent["app_day"]).between(1, 20)]
        recent = recent.drop(columns="app_day").drop_duplicates(["game_id", "team", "pitcher"])
        cand = pd.concat([cand, recent], ignore_index=True)
    cand = cand[cand["pitcher"] != cand["own_sp"]].reset_index(drop=True)
    if cand.empty:
        return pd.DataFrame(columns=["game_id", "team", "bp_woba", "bp_k", "bp_tired", "bp_left"])

    # Members of the starting rotation sit in the bullpen list too. Anyone who started
    # at least half of his last five outings is left out.
    role = pd.merge_asof(
        cand[["pitcher", "order"]].assign(_row=np.arange(len(cand))).sort_values("order", kind="stable"),
        apps[["pitcher", "order", "start_share"]].sort_values("order", kind="stable"),
        on="order", by="pitcher", allow_exact_matches=False).sort_values("_row")
    cand = cand[(role["start_share"].fillna(0.0) < 0.5).to_numpy()].reset_index(drop=True)

    totals = asof_totals(pa, cand, ["pitcher"], ["woba_num", "woba_den", "pa", "so"])
    cand = cand.merge(league["base"][["season", "lg_woba", "lg_k"]], on="season", how="left")
    cand["q"] = _shrunk(totals["woba_num"], totals["woba_den"], cand["lg_woba"], K_PIT).to_numpy()
    cand["k"] = _shrunk(totals["so"], totals["pa"], cand["lg_k"], K_PIT_K).to_numpy()

    daily = apps.groupby(["pitcher", "day"], as_index=False)[["pitches", "bf"]].sum()
    for back in (1, 2, 3):
        shifted = daily[["pitcher", "day", "pitches"]].rename(columns={"pitches": f"p{back}"})
        shifted["day"] = shifted["day"] + back
        cand = cand.merge(shifted, on=["pitcher", "day"], how="left")
        cand[f"p{back}"] = cand[f"p{back}"].fillna(0.0)
    daily = daily.sort_values("day", kind="stable")
    daily["cum_bf"] = daily.groupby("pitcher")["bf"].cumsum()
    cand["_row"] = np.arange(len(cand))
    recent_bf = {}
    for label, back in (("to", 1), ("from", 31)):
        probe = cand[["pitcher", "day", "_row"]].assign(day=cand["day"] - back).sort_values("day", kind="stable")
        got = pd.merge_asof(probe, daily[["pitcher", "day", "cum_bf"]], on="day", by="pitcher").sort_values("_row")
        recent_bf[label] = got["cum_bf"].fillna(0.0).to_numpy()
    cand["bf30"] = recent_bf["to"] - recent_bf["from"]

    available = np.ones(len(cand))
    p1, p2, p3 = cand["p1"].to_numpy(), cand["p2"].to_numpy(), cand["p3"].to_numpy()
    available = np.where(p1 > 0, 0.85, available)
    available = np.where(p1 >= 30, 0.40, available)
    available = np.where((p1 > 0) & (p2 > 0), 0.35, available)
    available = np.where((p1 > 0) & (p2 > 0) & (p3 > 0), 0.10, available)
    available = np.where(p1 + p2 + p3 >= 60, np.minimum(available, 0.30), available)
    cand["usage"] = cand["bf30"] + 3.0
    cand["ready"] = cand["usage"] * available
    cand["left"] = (cand["pitcher"].map(throws) == "L").astype(float)
    keys = [cand["game_id"], cand["team"]]
    ready = cand["ready"].groupby(keys).sum()
    out = pd.DataFrame({
        "bp_woba": (cand["q"] * cand["ready"]).groupby(keys).sum() / ready,
        "bp_k": (cand["k"] * cand["ready"]).groupby(keys).sum() / ready,
        "bp_tired": 1.0 - ready / cand["usage"].groupby(keys).sum(),
        "bp_left": (cand["left"] * cand["ready"]).groupby(keys).sum() / ready,
    }).reset_index()
    return out


# ---------------------------------------------------------------------------
# Defense and setting
# ---------------------------------------------------------------------------


def _defense(tg: pd.DataFrame, pa: pd.DataFrame, league: dict) -> pd.DataFrame:
    """Share of balls in play each team's fielders have turned into outs."""
    fielded = pa.rename(columns={"pit_team": "team"})
    queries = tg[["game_id", "team", "season", "order"]].copy()
    totals = asof_totals(fielded, queries, ["team"], ["bip", "bip_hit"])
    queries = queries.merge(league["base"][["season", "lg_bip_hit"]], on="season", how="left")
    queries["def_der"] = 1.0 - _shrunk(totals["bip_hit"], totals["bip"], queries["lg_bip_hit"], K_DEFENSE).to_numpy()
    return queries[["game_id", "team", "def_der"]]


def _setting(games: pd.DataFrame, extras: pd.DataFrame, league: dict) -> pd.DataFrame:
    """Ballpark scoring level, weather and the umpire's scoring tendency for each game."""
    out = games[["game_id", "season", "order", "home", "away", "status", "home_score", "away_score"]].merge(
        extras[["game_id", "venue", "temp", "wind_out", "roof_closed", "umpire"]], on="game_id", how="left")
    out["total"] = out["home_score"] + out["away_score"]
    played = out[(out["status"] == "final") & out["total"].notna()]

    # Ballpark: runs per game there, against runs per game in the same clubs' road games,
    # over the three seasons before this one.
    at_park = played.groupby(["venue", "season", "home"], as_index=False).agg(park_runs=("total", "sum"), park_n=("total", "size"))
    road = played.groupby(["season", "away"], as_index=False).agg(road_runs=("total", "sum"), road_n=("total", "size"))
    at_park = at_park.merge(road.rename(columns={"away": "home"}), on=["season", "home"], how="left")
    at_park["expected"] = at_park["park_n"] * at_park["road_runs"] / at_park["road_n"]
    rows = []
    for season in sorted(out["season"].unique()):
        window = at_park[(at_park["season"] < season) & (at_park["season"] >= season - 3)]
        if window.empty:
            continue
        agg = window.groupby("venue")[["park_runs", "expected", "park_n"]].sum()
        ratio = agg["park_runs"] / agg["expected"].replace(0.0, np.nan)
        factor = (ratio.fillna(1.0) * agg["park_n"] + K_PARK) / (agg["park_n"] + K_PARK)
        rows.append(pd.DataFrame({"venue": factor.index, "season": season, "park_rf": factor.to_numpy()}))
    park = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["venue", "season", "park_rf"])
    out = out.merge(park, on=["venue", "season"], how="left")
    out["park_rf"] = out["park_rf"].fillna(1.0)

    # Umpire: runs scored in his earlier games behind the plate, against the league.
    out = out.merge(league["base"][["season", "lg_runs"]], on="season", how="left")
    umpired = played[played["umpire"].notna()].assign(n=1.0)
    queries = out[["umpire", "season", "order"]].copy()
    queries["umpire"] = queries["umpire"].fillna(-1.0)
    totals = asof_totals(umpired, queries, ["umpire"], ["total", "n"], career=True)
    out["ump_runs"] = (_shrunk(totals["total"], totals["n"], 2.0 * out["lg_runs"], K_UMPIRE) - 2.0 * out["lg_runs"]).to_numpy()
    # Indoors the weather does not matter: a mild, still day stands in.
    indoors = out["roof_closed"].fillna(0.0) > 0
    out["temp"] = out["temp"].where(~indoors, 72.0)
    out["wind_out"] = out["wind_out"].where(~indoors, 0.0)
    return out[["game_id", "park_rf", "temp", "wind_out", "ump_runs"]]


# ---------------------------------------------------------------------------
# Statcast season tables and the injured list
# ---------------------------------------------------------------------------


def _statcast_prior(tables: dict, ids: "_Ids") -> dict[str, pd.DataFrame]:
    """Last season's Statcast numbers, keyed by the season they are used in.

    ``xgap`` is expected minus actual hitting (wOBA): above zero means a hitter
    was unlucky last year, or a pitcher was lucky. It counts in full only for
    a player with a full season of plate appearances.
    """
    empty = {"bat": pd.DataFrame(columns=["pid", "season", "xgap"]),
             "pit": pd.DataFrame(columns=["pid", "season", "xgap"]),
             "field": pd.DataFrame(columns=["pid", "season", "oaa"]),
             "speed": pd.DataFrame(columns=["pid", "season", "sprint_speed"])}
    for frame in empty.values():
        frame["pid"] = frame["pid"].astype("int64")
        frame["season"] = frame["season"].astype("int64")
    out = dict(empty)
    expected = tables.get("sc_expected")
    if expected is not None and len(expected):
        expected = expected.copy()
        expected["pid"] = ids(expected["player_id"])
        expected["season"] = expected["season"].astype("int64") + 1
        for role, full in (("bat", 300.0), ("pit", 400.0)):
            part = expected[expected["role"] == role]
            gap = (part["xwoba"] - part["woba"]) * (part["pa"] / full).clip(upper=1.0)
            out[role] = pd.DataFrame({"pid": part["pid"], "season": part["season"], "xgap": gap.fillna(0.0)}
                                     ).drop_duplicates(["pid", "season"])
    for key, name, column in (("field", "sc_fielding", "oaa"), ("speed", "sc_speed", "sprint_speed")):
        table = tables.get(name)
        if table is not None and len(table):
            table = table.copy()
            table["pid"] = ids(table["player_id"])
            table["season"] = table["season"].astype("int64") + 1
            out[key] = table[["pid", "season", column]].dropna().drop_duplicates(["pid", "season"])
    return out


def _injured(tg: pd.DataFrame, spells: pd.DataFrame | None, pa: pd.DataFrame, league: dict, ids: "_Ids") -> pd.DataFrame:
    """How much hitting and pitching each team has on the injured list before each game.

    Each injured player counts for his share of a regular's playing time
    multiplied by how much better than a stand-in he has been, both measured
    from games before he went on the list.
    """
    columns = ["game_id", "team", "il_bat", "il_pit", "il_n"]
    if spells is None or spells.empty:
        return pd.DataFrame(columns=columns)
    s = spells.copy()
    s["pid"] = ids(s["player_id"])
    s["season"] = s["start"].dt.year.astype("int64")
    s["order"] = order_key(s["start"], pd.Series(0, index=s.index))
    epoch = pd.Timestamp("2000-01-01")
    s["start_day"] = (s["start"] - epoch).dt.days
    s["end_day"] = (s["end"] - epoch).dt.days
    s = s[s["pid"] >= 0].reset_index(drop=True)
    if s.empty:
        return pd.DataFrame(columns=columns)
    as_hitter = asof_totals(pa, s.assign(batter=s["pid"]), ["batter"], ["woba_num", "woba_den", "pa"])
    as_pitcher = asof_totals(pa, s.assign(pitcher=s["pid"]), ["pitcher"], ["woba_num", "woba_den", "pa"])
    lg = s[["season"]].merge(league["base"][["season", "lg_woba"]], on="season", how="left")["lg_woba"].to_numpy()
    pitcher = (as_pitcher["pa"] > as_hitter["pa"]).to_numpy()
    hit = _shrunk(as_hitter["woba_num"], as_hitter["woba_den"], UNKNOWN_HITTER * lg, K_BAT).to_numpy()
    allowed = _shrunk(as_pitcher["woba_num"], as_pitcher["woba_den"], lg, K_PIT).to_numpy()
    bat_value = np.clip(as_hitter["pa"].to_numpy() / 900.0, 0.0, 1.0) * np.clip((hit - 0.88 * lg) / lg, 0.0, None)
    pit_value = np.clip(as_pitcher["pa"].to_numpy() / 900.0, 0.0, 1.0) * np.clip((1.10 * lg - allowed) / lg, 0.0, None)
    s["il_bat"] = np.where(pitcher, 0.0, bat_value)
    s["il_pit"] = np.where(pitcher, pit_value, 0.0)
    merged = tg[["game_id", "team", "season", "day"]].merge(
        s[["team", "season", "start_day", "end_day", "il_bat", "il_pit"]], on=["team", "season"])
    merged = merged[(merged["day"] >= merged["start_day"]) & (merged["day"] <= merged["end_day"])]
    out = merged.groupby(["game_id", "team"], as_index=False).agg(
        il_bat=("il_bat", "sum"), il_pit=("il_pit", "sum"), il_n=("il_bat", "size"))
    out["il_n"] = out["il_n"].astype(float)
    return out[columns]
