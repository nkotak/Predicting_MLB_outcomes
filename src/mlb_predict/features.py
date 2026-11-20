"""Model inputs, built so that a game can never see its own result.

Every number produced here for a game uses only games that were played
earlier (an earlier date, or the first game of the same day's doubleheader).
That holds for finished games and for games that have not been played, because
both go through exactly the same code.

The model is only ever given the columns named in ``FEATURES``. Adding a
column to a table somewhere else cannot leak into the model by accident.

Three kinds of input:

* team strength: record, runs scored and allowed, batting rates, bullpen,
  recent form and a running rating (Elo). Season-to-date numbers are blended
  with last season's, weighted by how many games have been played.
* starting pitcher: his own strikeout, walk, home run and run rates over the
  current and recent seasons, his last five starts, and days of rest.
  A pitcher with little or no history is pulled toward the league average.
* context: rest, doubleheader, day or night, and how far into the season it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PYTHAG_EXPONENT = 1.83
# How many games (or plate appearances, or outs) of "last season" a team or
# pitcher carries into the new season. Bigger means slower to react.
K_TEAM_GAMES = 40.0
K_TEAM_PA = 1500.0
K_PEN_OUTS = 450.0
K_PEN_BF = 600.0
K_SP = {"k": 150.0, "bb": 250.0, "hr": 500.0, "h": 400.0, "ra": 250.0}  # batters faced, or outs for "ra"
K_SP_STARTS = 5.0
# Weight given to a pitcher's earlier seasons: last year, two years ago, older.
SEASON_WEIGHTS = {1: 0.6, 2: 0.35}
OLDER_SEASON_WEIGHT = 0.2
BF_PER_INNING = 4.3

ELO_START = 1500.0
ELO_K = 4.0
ELO_HOME = 24.0
ELO_CARRY = 2.0 / 3.0  # share of a team's rating gap kept from one season to the next

TEAM_STATS = (
    "win_pct", "pythag", "rs_pg", "ra_pg", "ops", "iso", "bat_k", "bat_bb",
    "form10", "rd30", "pen_ra9", "pen_kbb", "pen_outs3", "rest", "elo",
)
SP_STATS = (
    "sp_k", "sp_bb", "sp_hr", "sp_ra9", "sp_fip", "sp_whip", "sp_outs_per_start",
    "sp_gs5", "sp_exp", "sp_rest",
)
CONTEXT = ("home_sp_known", "away_sp_known", "season_progress", "is_doubleheader", "is_night")

#: The only columns a model is allowed to see.
FEATURES: tuple[str, ...] = (
    tuple(f"d_{s}" for s in TEAM_STATS) + tuple(f"d_{s}" for s in SP_STATS) + CONTEXT
)

META = ("game_id", "date", "season", "number", "home", "away", "status",
        "home_sp_id", "away_sp_id", "home_score", "away_score", "home_win")


def order_key(date: pd.Series, number: pd.Series) -> pd.Series:
    """A sortable number for "when was this game played"."""
    return date.dt.strftime("%Y%m%d").astype("int64") * 10 + number.astype("int64")


# ---------------------------------------------------------------------------
# Running team rating
# ---------------------------------------------------------------------------


def elo_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's rating before every game.

    A team gains rating points when it wins and loses them when it loses, more
    so for an upset or a lopsided score. Between seasons every team drifts a
    third of the way back to average.
    """
    games = games.sort_values("order")
    rating: dict[str, float] = {}
    last_season: dict[str, int] = {}
    home_pre = np.empty(len(games))
    away_pre = np.empty(len(games))
    cols = zip(games["home"].to_numpy(), games["away"].to_numpy(), games["season"].to_numpy(),
               games["home_score"].to_numpy(dtype=float), games["away_score"].to_numpy(dtype=float),
               (games["status"] == "final").to_numpy())
    for i, (home, away, season, hs, as_, final) in enumerate(cols):
        for team in (home, away):
            if team not in rating:
                rating[team] = ELO_START
            elif last_season[team] != season:
                rating[team] = ELO_START + ELO_CARRY * (rating[team] - ELO_START)
            last_season[team] = season
        home_pre[i], away_pre[i] = rating[home], rating[away]
        if final and hs == hs and as_ == as_ and hs != as_:
            expected = 1.0 / (1.0 + 10.0 ** (-(rating[home] + ELO_HOME - rating[away]) / 400.0))
            margin = abs(hs - as_)
            step = ELO_K * min(np.sqrt(margin), 3.0) * ((1.0 if hs > as_ else 0.0) - expected)
            rating[home] += step
            rating[away] -= step
    return pd.DataFrame({"game_id": games["game_id"].to_numpy(), "home_elo": home_pre, "away_elo": away_pre})


# ---------------------------------------------------------------------------
# Team timeline
# ---------------------------------------------------------------------------

_TEAM_SUMS = ["played", "wins", "runs", "runs_allowed", "pa", "ab", "h", "tb", "bb", "hbp", "sf", "so",
              "hr", "pen_outs", "pen_r", "pen_bf", "pen_so", "pen_bb"]


def _blend(current_num, current_den, prior_rate, k):
    """Season-to-date rate pulled toward ``prior_rate`` with the weight of ``k`` units."""
    return (current_num + k * prior_rate) / (current_den + k)


def team_timeline(games: pd.DataFrame, team_games: pd.DataFrame, pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game holding that team's numbers *before* the game."""
    base = games[["game_id", "date", "season", "number", "order", "status", "home", "away"]]
    home = base.rename(columns={"home": "team", "away": "opp"}).assign(is_home=True)
    away = base.rename(columns={"away": "team", "home": "opp"}).assign(is_home=False)
    tl = pd.concat([home, away], ignore_index=True)

    stats = team_games.drop(columns=["is_home"]).copy()
    stats["tb"] = stats["h"] + stats["doubles"] + 2 * stats["triples"] + 3 * stats["hr"]
    pen = (
        pitcher_games[~pitcher_games["is_start"]]
        .groupby(["game_id", "team"], as_index=False)
        .agg(pen_outs=("outs", "sum"), pen_r=("r", "sum"), pen_bf=("bf", "sum"),
             pen_so=("so", "sum"), pen_bb=("bb", "sum"))
    )
    tl = tl.merge(stats, on=["game_id", "team"], how="left").merge(pen, on=["game_id", "team"], how="left")
    opp_runs = stats[["game_id", "team", "runs"]].rename(columns={"team": "opp", "runs": "runs_allowed"})
    tl = tl.merge(opp_runs, on=["game_id", "opp"], how="left")

    final = (tl["status"] == "final") & tl["runs"].notna()
    tl["played"] = final.astype(float)
    tl["wins"] = (final & (tl["runs"] > tl["runs_allowed"])).astype(float)
    pen_cols = ["pen_outs", "pen_r", "pen_bf", "pen_so", "pen_bb"]
    tl.loc[final, pen_cols] = tl.loc[final, pen_cols].fillna(0.0)  # a complete game has no relievers
    tl = tl.sort_values(["team", "order"]).reset_index(drop=True)

    filled = tl[_TEAM_SUMS].fillna(0.0)
    keys = [tl["team"], tl["season"]]
    cum = filled.groupby(keys).cumsum() - filled  # totals before this game, this season
    cum.columns = [f"cum_{c}" for c in cum.columns]

    # Last season's totals for the team, and for the league as a whole.
    season_tot = filled.groupby(keys).sum().reset_index()
    league_tot = season_tot.drop(columns=["team"]).groupby("season").sum().reset_index()
    prior_team = season_tot.assign(season=season_tot["season"] + 1)
    prior_team.columns = ["team", "season"] + [f"prev_{c}" for c in _TEAM_SUMS]
    prior_lg = league_tot.assign(season=league_tot["season"] + 1)
    prior_lg.columns = ["season"] + [f"lg_{c}" for c in _TEAM_SUMS]
    # The first season in the data has no "last season". It only warms up the
    # running numbers and is never used to train or test a model, so its own
    # league totals stand in.
    first = league_tot[league_tot["season"] == league_tot["season"].min()].copy()
    first.columns = ["season"] + [f"lg_{c}" for c in _TEAM_SUMS]
    prior_lg = pd.concat([first, prior_lg], ignore_index=True)

    tl = pd.concat([tl, cum], axis=1).merge(prior_team, on=["team", "season"], how="left")
    tl = tl.merge(prior_lg, on="season", how="left")

    def ratio(prefix: str, num: str, den: str) -> pd.Series:
        return tl[f"{prefix}_{num}"] / tl[f"{prefix}_{den}"].replace(0.0, np.nan)

    def prior(num: str, den: str) -> pd.Series:
        """Last season's team rate, pulled halfway to the league rate."""
        league = ratio("lg", num, den)
        team = ratio("prev", num, den)
        return (0.5 * team + 0.5 * league).fillna(league)

    out = tl[["game_id", "team", "is_home", "order", "date", "season"]].copy()
    out["gp"] = tl["cum_played"]
    out["win_pct"] = _blend(tl["cum_wins"], tl["cum_played"], prior("wins", "played"), K_TEAM_GAMES)
    out["rs_pg"] = _blend(tl["cum_runs"], tl["cum_played"], prior("runs", "played"), K_TEAM_GAMES)
    out["ra_pg"] = _blend(tl["cum_runs_allowed"], tl["cum_played"], prior("runs_allowed", "played"), K_TEAM_GAMES)
    rs, ra = out["rs_pg"] ** PYTHAG_EXPONENT, out["ra_pg"] ** PYTHAG_EXPONENT
    out["pythag"] = rs / (rs + ra)

    def obp_parts(prefix: str):
        on = tl[f"{prefix}_h"] + tl[f"{prefix}_bb"] + tl[f"{prefix}_hbp"]
        chances = tl[f"{prefix}_ab"] + tl[f"{prefix}_bb"] + tl[f"{prefix}_hbp"] + tl[f"{prefix}_sf"]
        return on, chances.replace(0.0, np.nan)

    def prior_obp() -> pd.Series:
        l_on, l_ch = obp_parts("lg")
        p_on, p_ch = obp_parts("prev")
        league = l_on / l_ch
        return (0.5 * (p_on / p_ch) + 0.5 * league).fillna(league)

    c_on, _ = obp_parts("cum")
    c_chances = tl["cum_ab"] + tl["cum_bb"] + tl["cum_hbp"] + tl["cum_sf"]
    obp = _blend(c_on, c_chances, prior_obp(), K_TEAM_PA)
    slg = _blend(tl["cum_tb"], tl["cum_ab"], prior("tb", "ab"), K_TEAM_PA)
    avg = _blend(tl["cum_h"], tl["cum_ab"], prior("h", "ab"), K_TEAM_PA)
    out["ops"] = obp + slg
    out["iso"] = slg - avg
    out["bat_k"] = _blend(tl["cum_so"], tl["cum_pa"], prior("so", "pa"), K_TEAM_PA)
    out["bat_bb"] = _blend(tl["cum_bb"], tl["cum_pa"], prior("bb", "pa"), K_TEAM_PA)

    out["pen_ra9"] = 27.0 * _blend(tl["cum_pen_r"], tl["cum_pen_outs"], prior("pen_r", "pen_outs"), K_PEN_OUTS)
    pen_k = _blend(tl["cum_pen_so"], tl["cum_pen_bf"], prior("pen_so", "pen_bf"), K_PEN_BF)
    pen_w = _blend(tl["cum_pen_bb"], tl["cum_pen_bf"], prior("pen_bb", "pen_bf"), K_PEN_BF)
    out["pen_kbb"] = pen_k - pen_w

    # Recent form inside the current season, each pulled toward neutral.
    by_team_season = [tl["team"], tl["season"]]
    won = tl["wins"].where(final)
    diff = (tl["runs"] - tl["runs_allowed"]).where(final)

    def before(series: pd.Series, window: int, how: str) -> pd.Series:
        def roll(s: pd.Series) -> pd.Series:
            r = s.shift(1).rolling(window, min_periods=1)
            return r.sum() if how == "sum" else r.count()
        return series.groupby(by_team_season).transform(roll).fillna(0.0)

    out["form10"] = (before(won, 10, "sum") + 2.5) / (before(won, 10, "count") + 5.0)
    out["rd30"] = before(diff, 30, "sum") / (before(diff, 30, "count") + 10.0)

    # Days since the team's previous game, and bullpen work over the previous three days.
    day = (tl["date"] - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    rest = np.full(len(tl), 5.0)
    pen3 = np.zeros(len(tl))
    pen_outs = tl["pen_outs"].fillna(0.0).to_numpy()
    played = final.to_numpy()
    team_codes = tl["team"].to_numpy()
    start = 0
    for end in range(1, len(tl) + 1):
        if end == len(tl) or team_codes[end] != team_codes[start]:
            d = day[start:end]
            p = np.where(played[start:end], pen_outs[start:end], 0.0)
            prefix = np.concatenate([[0.0], np.cumsum(p)])
            left = np.searchsorted(d, d - 3, side="left")
            idx = np.arange(end - start)
            pen3[start:end] = prefix[idx] - prefix[np.minimum(left, idx)]
            gap = np.diff(d, prepend=d[0] - 5).astype(float)
            rest[start:end] = np.clip(gap, 0.0, 5.0)
            start = end
    out["rest"] = rest
    out["pen_outs3"] = pen3
    return out


# ---------------------------------------------------------------------------
# Starting pitchers
# ---------------------------------------------------------------------------

_SP_SUMS = ["g", "gs", "outs", "bf", "h", "hr", "r", "bb", "hbp", "so", "start_outs"]


def _season_weight(gap: np.ndarray) -> np.ndarray:
    weights = np.full(gap.shape, OLDER_SEASON_WEIGHT)
    for years, weight in SEASON_WEIGHTS.items():
        weights[gap == years] = weight
    return weights


def starter_features(queries: pd.DataFrame, pitcher_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each queried pitcher's record before the queried game.

    ``queries`` has one row per (game, pitcher): ``game_id``, ``player_id``,
    ``season`` and ``order``. A missing ``player_id`` (starter not announced)
    gets league-average numbers and is flagged as unknown.
    """
    # Join keys must have one type on both sides whatever pandas version is in
    # use, so ids are plain Python strings and seasons are 64-bit integers.
    queries = queries.copy()
    queries["player_id"] = queries["player_id"].astype(object)
    queries["season"] = queries["season"].astype("int64")
    when = games[["game_id", "order", "season", "date"]]
    app = pitcher_games.assign(player_id=pitcher_games["player_id"].astype(object))
    app = app.merge(when, on="game_id", how="inner")
    app["season"] = app["season"].astype("int64")
    app["g"] = 1.0
    app["gs"] = app["is_start"].astype(float)
    app["start_outs"] = app["outs"].where(app["is_start"], 0.0)
    # A simple "how good was this start" number: outs and strikeouts add,
    # hits, walks, runs and home runs subtract. 50 is an average start.
    app["game_score"] = (40.0 + 2.0 * app["outs"] + app["so"] - 2.0 * app["bb"]
                         - 2.0 * app["h"] - 3.0 * app["r"] - 6.0 * app["hr"])
    app = app.sort_values(["player_id", "order"]).reset_index(drop=True)
    sums = app[_SP_SUMS].fillna(0.0)

    # League rates by season, used as "what an unknown pitcher looks like".
    lg = sums.groupby(app["season"]).sum()
    league = pd.DataFrame({
        "lg_k": lg["so"] / lg["bf"], "lg_bb": (lg["bb"] + lg["hbp"]) / lg["bf"], "lg_hr": lg["hr"] / lg["bf"],
        "lg_h": lg["h"] / lg["bf"], "lg_ra": lg["r"] / lg["outs"], "lg_start_outs": lg["start_outs"] / lg["gs"],
    })
    league_prior = league.copy()
    league_prior.index = league_prior.index + 1
    league_prior = pd.concat([league.loc[[league.index.min()]], league_prior])
    league_prior = league_prior[~league_prior.index.duplicated(keep="last")].reset_index(names="season")

    # Earlier seasons, weighted so that older ones count for less.
    season_tot = sums.groupby([app["player_id"], app["season"]]).sum().reset_index()
    pairs = queries.loc[queries["player_id"].notna(), ["player_id", "season"]].drop_duplicates()
    earlier = pairs.merge(season_tot, on="player_id", suffixes=("", "_past"))
    earlier = earlier[earlier["season_past"] < earlier["season"]]
    weights = _season_weight((earlier["season"] - earlier["season_past"]).to_numpy())
    weighted = earlier[_SP_SUMS].mul(weights, axis=0)
    weighted[["player_id", "season"]] = earlier[["player_id", "season"]]
    past = weighted.groupby(["player_id", "season"], as_index=False).sum()
    past.columns = ["player_id", "season"] + [f"past_{c}" for c in _SP_SUMS]

    # This season's totals after each appearance. merge_asof below then picks
    # the last appearance strictly before the queried game.
    running = sums.groupby([app["player_id"], app["season"]]).cumsum()
    running.columns = [f"cur_{c}" for c in _SP_SUMS]
    running[["player_id", "season", "order"]] = app[["player_id", "season", "order"]]

    q = queries.copy()
    q["row"] = np.arange(len(q))
    known = q[q["player_id"].notna()].sort_values("order")
    known = pd.merge_asof(known, running.sort_values("order"), on="order", by=["player_id", "season"],
                          allow_exact_matches=False)
    known = known.merge(past, on=["player_id", "season"], how="left")

    # Last appearance of any kind (for rest) and last five starts (for form).
    last_any = app[["player_id", "order", "date"]].rename(columns={"date": "last_date"})
    known = pd.merge_asof(known.sort_values("order"), last_any.sort_values("order"), on="order",
                          by="player_id", allow_exact_matches=False)
    starts = app[app["is_start"]].copy()
    grp = starts.groupby("player_id")["game_score"]
    starts["gs5_sum"] = grp.transform(lambda s: s.rolling(5, min_periods=1).sum())
    starts["gs5_n"] = grp.transform(lambda s: s.rolling(5, min_periods=1).count())
    known = pd.merge_asof(known.sort_values("order"),
                          starts[["player_id", "order", "gs5_sum", "gs5_n"]].sort_values("order"),
                          on="order", by="player_id", allow_exact_matches=False)

    q = q.merge(known.drop(columns=["game_id", "player_id", "season", "order"]), on="row", how="left")
    for column in ["last_date", "gs5_sum", "gs5_n", *[f"cur_{c}" for c in _SP_SUMS], *[f"past_{c}" for c in _SP_SUMS]]:
        if column not in q.columns:
            q[column] = pd.NaT if column == "last_date" else np.nan
    q = q.merge(league_prior, on="season", how="left").sort_values("row")
    tot = {c: q[f"cur_{c}"].fillna(0.0) + q[f"past_{c}"].fillna(0.0) for c in _SP_SUMS}

    out = pd.DataFrame({"game_id": q["game_id"].to_numpy()})
    k = (tot["so"] + K_SP["k"] * q["lg_k"]) / (tot["bf"] + K_SP["k"])
    bb = (tot["bb"] + tot["hbp"] + K_SP["bb"] * q["lg_bb"]) / (tot["bf"] + K_SP["bb"])
    hr = (tot["hr"] + K_SP["hr"] * q["lg_hr"]) / (tot["bf"] + K_SP["hr"])
    hits = (tot["h"] + K_SP["h"] * q["lg_h"]) / (tot["bf"] + K_SP["h"])
    ra = (tot["r"] + K_SP["ra"] * q["lg_ra"]) / (tot["outs"] + K_SP["ra"])
    out["sp_k"] = k.to_numpy()
    out["sp_bb"] = bb.to_numpy()
    out["sp_hr"] = hr.to_numpy()
    out["sp_ra9"] = (27.0 * ra).to_numpy()
    out["sp_fip"] = ((13.0 * hr + 3.0 * bb - 2.0 * k) * BF_PER_INNING + 3.1).to_numpy()
    out["sp_whip"] = ((hits + bb) * BF_PER_INNING).to_numpy()
    out["sp_outs_per_start"] = (
        (tot["start_outs"] + K_SP_STARTS * q["lg_start_outs"]) / (tot["gs"] + K_SP_STARTS)
    ).to_numpy()
    out["sp_gs5"] = ((q["gs5_sum"].fillna(0.0) + 2.0 * 50.0) / (q["gs5_n"].fillna(0.0) + 2.0)).to_numpy()
    out["sp_exp"] = np.log1p(tot["bf"]).to_numpy()
    query_date = q["game_id"].map(games.set_index("game_id")["date"])
    rest = (query_date - q["last_date"]).dt.days
    out["sp_rest"] = rest.clip(lower=0, upper=10).fillna(10.0).to_numpy()
    out["sp_known"] = (tot["bf"] > 0).astype(float).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def build_features(games: pd.DataFrame, team_games: pd.DataFrame, pitcher_games: pd.DataFrame) -> pd.DataFrame:
    """One row per game with the model inputs and a few descriptive columns.

    Works for finished and scheduled games alike. ``home_win`` is 1 or 0 for a
    finished game and empty for a tie or a game that has not been played.
    """
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    games["order"] = order_key(games["date"], games["number"])
    games = games.sort_values("order").reset_index(drop=True)

    timeline = team_timeline(games, team_games, pitcher_games)
    elo = elo_ratings(games)

    frame = games[["game_id", "date", "season", "number", "home", "away", "status", "order",
                   "home_sp_id", "away_sp_id", "home_score", "away_score", "day_night"]].copy()
    decided = (frame["status"] == "final") & (frame["home_score"] != frame["away_score"])
    frame["home_win"] = np.where(decided, (frame["home_score"] > frame["away_score"]).astype(float), np.nan)

    team_cols = [s for s in TEAM_STATS if s != "elo"] + ["gp"]
    for side, flag in (("home", True), ("away", False)):
        part = timeline.loc[timeline["is_home"] == flag, ["game_id", *team_cols]]
        part = part.rename(columns={c: f"{side}_{c}" for c in team_cols})
        frame = frame.merge(part, on="game_id", how="left")
    frame = frame.merge(elo, on="game_id", how="left")

    for side in ("home", "away"):
        queries = frame[["game_id", f"{side}_sp_id", "season", "order"]].rename(
            columns={f"{side}_sp_id": "player_id"})
        sp = starter_features(queries, pitcher_games, games)
        sp = sp.rename(columns={c: f"{side}_{c}" for c in sp.columns if c != "game_id"})
        frame = frame.merge(sp, on="game_id", how="left")

    for stat in TEAM_STATS + SP_STATS:
        frame[f"d_{stat}"] = frame[f"home_{stat}"] - frame[f"away_{stat}"]
    frame["season_progress"] = (frame["home_gp"] / 162.0).clip(0.0, 1.0)
    frame["is_doubleheader"] = (frame["number"] > 0).astype(float)
    frame["is_night"] = (frame["day_night"] == "N").astype(float)
    return frame.sort_values("order").reset_index(drop=True)


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns a model may see, and nothing else."""
    return frame.loc[:, list(FEATURES)].astype(float)
