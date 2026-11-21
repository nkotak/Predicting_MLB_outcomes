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
    tables = {"games": games, "team_games": pd.DataFrame(team_rows),
              "pitcher_games": pd.DataFrame(pitcher_rows), "players": players}
    tables.update(make_player_tables(tables, seed=seed + 100))
    return tables


PA_SUMS = ("pa", "ab", "h1", "h2", "h3", "hr", "bb", "ibb", "hbp", "so", "sf", "woba_num", "woba_den",
           "pitches", "whiffs", "called", "swings", "fb_speed_sum", "fb_n",
           "bip", "bip_hit", "tracked", "gb", "ev_sum", "ev_n", "hard")


def make_player_tables(tables, seed: int = 100) -> dict:
    """Made-up lineups, rosters and plate appearances that fit the made-up games.

    Each club has eleven hitters with their own skill and batting side. Nine
    start, and now and then a regular rests. Every starter in the lineup bats
    three times against the opposing starter and once against a reliever.
    The last scheduled game has no lineup posted yet.
    """
    rng = np.random.default_rng(seed)
    games, pitcher_games = tables["games"], tables["pitcher_games"]
    hitters = {t: [f"{2000 + 20 * i + j}" for j in range(11)] for i, t in enumerate(TEAMS)}
    bat_skill = {h: rng.normal(0, 0.035) for t in TEAMS for h in hitters[t]}
    bat_side = {h: ("L", "R", "R", "S")[k % 4] for t in TEAMS for k, h in enumerate(hitters[t])}
    used = {(r.game_id, r.team): [] for r in pitcher_games.itertuples()}
    for r in pitcher_games.sort_values("seq").itertuples():
        used[(r.game_id, r.team)].append(r.player_id)
    all_pitchers = pitcher_games.groupby("team")["player_id"].unique().to_dict()
    # Every fifth-slot starter throws left-handed, so both hands appear.
    throws = {p: ("L" if p.endswith("4") else "R") for ps in all_pitchers.values() for p in ps}
    pit_skill = {p: rng.normal(0, 0.03) for p in throws}
    # Left- and right-handers are made equally good on average (starters and
    # relievers separately), so that any gap between the two in the data comes
    # from the left/right matchup alone.
    for role in ("1", "5"):
        group = [p for p in pit_skill if p.startswith(role)]
        gap = np.mean([pit_skill[p] for p in group if throws[p] == "L"]) - np.mean(
            [pit_skill[p] for p in group if throws[p] == "R"])
        for p in group:
            if throws[p] == "L":
                pit_skill[p] -= gap

    lineup_rows, roster_rows, extra_rows = [], [], []
    col = {k: [] for k in ("game_id", "bat_team", "pit_team", "batter", "pitcher", "vs_starter", "times")}
    scheduled_ids = list(games.loc[games["status"] == "scheduled", "game_id"])
    for game in games.itertuples(index=False):
        final = game.status == "final"
        extra_rows.append({"game_id": game.game_id, "venue": 100 + TEAMS.index(game.home),
                           "temp": float(rng.integers(50, 95)), "wind_mph": float(rng.integers(0, 15)),
                           "wind_out": float(rng.integers(-10, 11)), "roof_closed": 0.0, "sky": "Clear",
                           "umpire": float(900 + rng.integers(0, 6))})
        for team, opp, opp_sp in ((game.home, game.away, game.away_sp_id), (game.away, game.home, game.home_sp_id)):
            squad = hitters[team]
            nine = list(squad[:9])
            if rng.random() < 0.3:  # a regular gets the day off
                nine[int(rng.integers(0, 9))] = squad[9 + int(rng.integers(0, 2))]
            bench = [h for h in squad if h not in nine]
            posted = final or game.game_id != scheduled_ids[-1]
            if posted:
                lineup_rows += [(game.game_id, team, k + 1, h, "DH" if k == 8 else "1B") for k, h in enumerate(nine)]
            pitched = used.get((game.game_id, team), [])
            roster_rows += [(game.game_id, team, h, "BAT") for h in nine]
            roster_rows += [(game.game_id, team, h, "BN") for h in bench]
            for pitcher in all_pitchers.get(team, []):
                role = "BP" if pitcher not in pitched else ("SP" if pitcher == pitched[0] else "RP")
                roster_rows.append((game.game_id, team, pitcher, role))
            if not final:
                continue
            for pitcher, times, vs_starter in ((opp_sp, 3, True), (used[(game.game_id, opp)][1], 1, False)):
                col["game_id"] += [game.game_id] * 9
                col["bat_team"] += [team] * 9
                col["pit_team"] += [opp] * 9
                col["batter"] += nine
                col["pitcher"] += [pitcher] * 9
                col["vs_starter"] += [vs_starter] * 9
                col["times"] += [times] * 9

    # Outcomes for every plate appearance at once.
    plate = pd.DataFrame(col)
    n = len(plate)
    times = plate["times"].to_numpy()
    hand = plate["pitcher"].map(throws).to_numpy()
    natural = plate["batter"].map(bat_side).to_numpy()
    side = np.where(natural == "S", np.where(hand == "L", "R", "L"), natural)
    pitcher_skill = plate["pitcher"].map(pit_skill).to_numpy()
    edge = plate["batter"].map(bat_skill).to_numpy() - pitcher_skill + np.where(side != hand, 0.012, -0.012)
    live = np.arange(3)[None, :] < times[:, None]
    draws, kind = rng.random((n, 3)), rng.random((n, 3))
    on_base = np.clip(0.32 + edge, 0.05, 0.6)[:, None]
    hit = live & (draws < on_base * 0.72)
    walk = live & (draws >= on_base * 0.72) & (draws < on_base)
    out = live & ~hit & ~walk

    def count(mask):
        return mask.sum(axis=1).astype(float)

    plate["bat_side"], plate["pit_hand"] = side, hand
    plate["pa"] = times.astype(float)
    plate["ab"] = count(hit | out)
    plate["h1"] = count(hit & (kind < 0.68))
    plate["h2"] = count(hit & (kind >= 0.68) & (kind < 0.88))
    plate["h3"] = 0.0
    plate["hr"] = count(hit & (kind >= 0.88))
    plate["bb"] = count(walk)
    plate["ibb"] = plate["hbp"] = plate["sf"] = 0.0
    plate["so"] = count(out & (kind < 0.30))
    plate["woba_num"] = 0.69 * plate["bb"] + 0.88 * plate["h1"] + 1.25 * plate["h2"] + 2.03 * plate["hr"]
    plate["woba_den"] = plate["ab"] + plate["bb"]
    plate["pitches"] = 4.0 * times
    plate["whiffs"] = rng.binomial(4 * times, np.clip(0.10 + pitcher_skill, 0.02, 0.3)).astype(float)
    plate["called"] = times.astype(float)
    plate["swings"] = 2.0 * times
    plate["fb_n"] = 2.0 * times
    plate["fb_speed_sum"] = plate["fb_n"] * (92.0 + 30.0 * pitcher_skill + rng.normal(0, 0.8, n))
    plate["bip"] = (plate["pa"] - plate["bb"] - plate["so"] - plate["hr"]).clip(lower=0.0)
    plate["bip_hit"] = plate["h1"] + plate["h2"]
    plate["tracked"] = plate["bip"]
    plate["gb"] = (plate["bip"] * rng.random(n) < 0.45 * plate["bip"]).astype(float)
    plate["ev_n"] = plate["bip"]
    plate["hard"] = (rng.random(n) < 0.4).astype(float) * (plate["bip"] > 0)
    plate["ev_sum"] = plate["ev_n"] * 89.0

    everyone = [(h, f"Hitter {h}", bat_side[h], "R", "1B") for t in TEAMS for h in hitters[t]]
    everyone += [(p, f"Pitcher {p}", "R", throws[p], "P") for p in throws]
    people = pd.DataFrame(everyone, columns=["player_id", "name", "bats", "throws", "position"])
    people["birth_date"] = [f"{1978 + (int(pid) * 7) % 12}-0{1 + int(pid) % 9}-15" for pid in people["player_id"]]
    seasons = sorted(int(s) for s in games["season"].unique())
    spells = pd.DataFrame(
        [{"player_id": hitters[t][10], "team": t, "start": pd.Timestamp(season, 4, 10), "end": pd.Timestamp(season, 4, 25)}
         for season in seasons for t in TEAMS[:4]]
        + [{"player_id": all_pitchers[t][0], "team": t, "start": pd.Timestamp(season, 5, 1), "end": pd.Timestamp(season, 5, 12)}
           for season in seasons for t in TEAMS[4:6] if len(all_pitchers.get(t, []))])
    spells["player_id"] = spells["player_id"].astype(object)
    expected = pd.DataFrame(
        [{"player_id": h, "season": season, "role": "bat", "pa": 500.0, "woba": 0.320, "xwoba": 0.320 + bat_skill[h] / 2}
         for season in seasons for t in TEAMS for h in hitters[t]]
        + [{"player_id": q, "season": season, "role": "pit", "pa": 600.0, "woba": 0.320, "xwoba": 0.320 - pit_skill[q] / 2}
           for season in seasons for q in throws])
    fielding = pd.DataFrame([{"player_id": h, "season": season, "oaa": float(int(100 * bat_skill[h]))}
                             for season in seasons for t in TEAMS for h in hitters[t]])
    speed = pd.DataFrame([{"player_id": h, "season": season, "sprint_speed": 27.0 + 20 * bat_skill[h]}
                          for season in seasons for t in TEAMS for h in hitters[t]])
    for frame in (expected, fielding, speed, people):
        frame["player_id"] = frame["player_id"].astype(object)
    lineups = pd.DataFrame(lineup_rows, columns=["game_id", "team", "slot", "player_id", "position"])
    rosters = pd.DataFrame(roster_rows, columns=["game_id", "team", "player_id", "role"])
    for frame, column in ((lineups, "player_id"), (rosters, "player_id"), (plate, "batter"), (plate, "pitcher")):
        frame[column] = frame[column].astype(object)
    return {"lineups": lineups, "rosters": rosters,
            "plate_appearances": plate[["game_id", "bat_team", "pit_team", "batter", "pitcher", "bat_side",
                                        "pit_hand", "vs_starter", *PA_SUMS]],
            "game_extras": pd.DataFrame(extra_rows), "people": people, "il_spells": spells,
            "sc_expected": expected, "sc_fielding": fielding, "sc_speed": speed}


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
        # The best price among the books is a little better than the middle one.
        "best_home_dec": 1.0 / (fair * (1 + cut)) * 1.012, "best_away_dec": 1.0 / ((1 - fair) * (1 + cut)) * 1.012,
        "open_home_dec": 1.0 / (opening * (1 + cut)), "open_away_dec": 1.0 / ((1 - opening) * (1 + cut)),
        "open_market_home_prob": opening, "odds_fetched_utc": "2021-01-01T00:00:00Z",
    })
