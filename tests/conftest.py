"""A small made-up league used by the tests.

Nothing here touches the network. The league has real team codes, latent team
and pitcher strengths (so there is something for a model to learn), box scores
that add up, a doubleheader, and a few games that have not been played yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TEAMS = ["NYY", "BOS", "LAD", "SF", "CHC", "STL", "HOU", "ATL"]


def make_league(seed: int = 0, seasons=(2015, 2016, 2017), days: int = 60, scheduled: int = 4):
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 0.45) for t in TEAMS}
    starters = {t: [f"{1000 + 10 * i + j}" for j in range(5)] for i, t in enumerate(TEAMS)}
    relievers = {t: [f"{5000 + 10 * i + j}" for j in range(6)] for i, t in enumerate(TEAMS)}
    skill = {p: rng.normal(0, 0.5) for t in TEAMS for p in starters[t]}
    turn = {t: 0 for t in TEAMS}
    games, team_rows, pitcher_rows = [], [], []

    def play(date, home, away, number, final=True):
        game_id = f"{date:%Y%m%d}-{home}-{number}"
        home_sp, away_sp = starters[home][turn[home] % 5], starters[away][turn[away] % 5]
        turn[home] += 1
        turn[away] += 1
        row = {"game_id": game_id, "date": pd.Timestamp(date), "season": date.year, "number": number,
               "home": home, "away": away, "home_score": np.nan, "away_score": np.nan,
               "status": "final" if final else "scheduled", "day_night": rng.choice(["D", "N"]),
               "park": home, "source": "test", "mlb_game_pk": np.nan,
               "home_sp_id": home_sp, "away_sp_id": away_sp}
        if not final:
            games.append(row)
            return
        mean_home = np.exp(1.45 + 0.25 * (strength[home] - strength[away]) - 0.2 * skill[away_sp] + 0.08)
        mean_away = np.exp(1.45 + 0.25 * (strength[away] - strength[home]) - 0.2 * skill[home_sp])
        runs = {home: int(rng.poisson(mean_home)), away: int(rng.poisson(mean_away))}
        while runs[home] == runs[away]:
            runs[rng.choice([home, away])] += 1
        row["home_score"], row["away_score"] = runs[home], runs[away]
        games.append(row)
        for team, opp, is_home, sp in ((home, away, True, home_sp), (away, home, False, away_sp)):
            hits = runs[team] + int(rng.poisson(5))
            hr = min(runs[team], int(rng.poisson(1.0)))
            doubles, triples = int(rng.poisson(1.6)), int(rng.random() < 0.15)
            hits = max(hits, hr + doubles + triples)
            bb, hbp, sf, so = int(rng.poisson(3)), int(rng.random() < 0.3), int(rng.random() < 0.3), int(rng.poisson(8))
            ab = 27 + hits - 2
            allowed = runs[opp]
            sp_outs = int(np.clip(rng.normal(17 + 2 * skill[sp], 3), 6, 27))
            sp_runs = int(min(allowed, rng.binomial(allowed, 0.65))) if allowed else 0
            team_rows.append({
                "game_id": game_id, "team": team, "is_home": is_home, "pa": ab + bb + hbp + sf, "ab": ab,
                "runs": runs[team], "h": hits, "doubles": doubles, "triples": triples, "hr": hr, "bb": bb,
                "ibb": 0, "hbp": hbp, "so": so, "sf": sf, "sb": 0, "cs": 0, "lob": 7, "p_outs": 27,
                "p_bf": 38, "p_h": 8, "p_hr": 1, "p_r": allowed, "p_er": allowed, "p_bb": 3, "p_so": 8,
                "p_hbp": 0, "errors": 0,
            })
            pitcher_rows.append({
                "game_id": game_id, "team": team, "player_id": sp, "seq": 1, "is_start": True,
                "outs": sp_outs, "bf": sp_outs + 6, "h": int(rng.poisson(5)), "hr": int(rng.random() < 0.5),
                "r": sp_runs, "er": sp_runs, "bb": int(rng.poisson(2)), "ibb": 0,
                "so": int(min(rng.poisson(max(0.5, 5 + 1.5 * skill[sp])), 15)), "hbp": 0,
            })
            left, used = 27 - sp_outs, 0
            for seq, reliever in enumerate(rng.permutation(relievers[team])[:2], start=2):
                outs = left if seq == 3 else left // 2
                pitcher_rows.append({
                    "game_id": game_id, "team": team, "player_id": reliever, "seq": seq, "is_start": False,
                    "outs": outs, "bf": outs + 2, "h": 1, "hr": 0,
                    "r": (allowed - sp_runs) if seq == 3 else 0, "er": 0, "bb": int(rng.poisson(0.8)),
                    "ibb": 0, "so": int(rng.poisson(1.5)), "hbp": 0,
                })
                used += outs

    last_day = None
    for season in seasons:
        for day in range(days):
            date = pd.Timestamp(season, 4, 1) + pd.Timedelta(days=day)
            order = list(rng.permutation(TEAMS))
            for i in range(0, len(TEAMS), 2):
                last_pair = i == len(TEAMS) - 2
                if day != 20 and not last_pair and rng.random() < 0.12:
                    continue  # a day off for these two teams
                play(date, order[i], order[i + 1], 0)
            if day == 20:  # a doubleheader: the same two teams play twice
                games.pop(); team_rows[-2:] = []; del pitcher_rows[-6:]
                turn[order[-2]] -= 1; turn[order[-1]] -= 1
                play(date, order[-2], order[-1], 1)
                play(date, order[-2], order[-1], 2)
            last_day = date
    upcoming = last_day + pd.Timedelta(days=1)
    order = list(rng.permutation(TEAMS))
    for i in range(0, 2 * scheduled, 2):
        play(upcoming, order[i % 8], order[(i + 1) % 8], 0, final=False)
    games = pd.DataFrame(games)
    games["home_sp_id"] = games["home_sp_id"].astype(object)
    games["away_sp_id"] = games["away_sp_id"].astype(object)
    if scheduled:
        games.loc[games.index[-1], "home_sp_id"] = None  # one starter not announced yet
    players = pd.DataFrame({"player_id": [p for t in TEAMS for p in starters[t] + relievers[t]]})
    players["name"] = "Pitcher " + players["player_id"]
    players["throws"] = "R"
    return {"games": games, "team_games": pd.DataFrame(team_rows),
            "pitcher_games": pd.DataFrame(pitcher_rows), "players": players}


@pytest.fixture(scope="session")
def league():
    return make_league()


@pytest.fixture(scope="session")
def big_league():
    """Enough seasons and games to fit and test the model."""
    return make_league(seed=3, seasons=tuple(range(2012, 2022)), days=230)


def make_odds(tables, seed: int = 9, noise: float = 0.03, cut: float = 0.04) -> pd.DataFrame:
    """Made-up closing prices: a sensible market with a little noise and a bookmaker's cut."""
    from mlb_predict.features import build_features

    rng = np.random.default_rng(seed)
    frame = build_features(tables["games"], tables["team_games"], tables["pitcher_games"])
    fair = 1.0 / (1.0 + 10.0 ** (-(frame["d_elo"] + 24.0) / 400.0))
    fair = np.clip(fair + 0.15 * frame["d_sp_k"].fillna(0) * 4 + rng.normal(0, noise, len(frame)), 0.2, 0.8)
    opening = np.clip(fair + rng.normal(0, 0.025, len(frame)), 0.2, 0.8)  # the early price is a bit noisier
    return pd.DataFrame({
        "game_id": frame["game_id"], "event_id": [str(i) for i in range(len(frame))],
        "home_dec": 1.0 / (fair * (1 + cut)), "away_dec": 1.0 / ((1 - fair) * (1 + cut)),
        "market_home_prob": fair, "n_books": 5,
        "open_home_dec": 1.0 / (opening * (1 + cut)), "open_away_dec": 1.0 / ((1 - opening) * (1 + cut)),
        "open_market_home_prob": opening, "odds_fetched_utc": "2021-01-01T00:00:00Z",
    })
