"""Everything the model is given for a game: the team view plus the player view.

``features.py`` describes each club as a whole and each starting pitcher from
his box score lines. ``players.py`` looks at the individual hitters, the
left/right matchups, the bullpen and the setting. This module joins the two
and names the columns the model is allowed to read.
"""

from __future__ import annotations

import pandas as pd

from . import features, players

#: Groups of player-level columns that earned their place in the
#: season-by-season tests. What each one was worth is in reports/BACKTEST.md.
GROUPS_USED: tuple[str, ...] = ("lineup", "history", "starter_stuff", "defense", "statcast", "injuries", "age",
                                "expected_runs")
#: Groups that were built and tested and did not improve the forecasts when
#: added to the ones above:
#:   left_right   as columns of their own. The left/right matchup still reaches
#:                the model inside expected runs.
#:   bullpen      reliever-by-reliever quality and tiredness. The team view
#:                already has bullpen results and recent workload, and the
#:                available bullpen is part of expected runs.
#:   recent_form  hot and cold streaks over three weeks
#:   setting      ballpark, weather and umpire. They change how many runs are
#:                scored, not who scores more. The ballpark is part of expected runs.
GROUPS_LEFT_OUT: tuple[str, ...] = tuple(g for g in players.GROUPS if g not in GROUPS_USED)

PLAYER_INPUTS: tuple[str, ...] = tuple(column for group in GROUPS_USED for column in players.GROUPS[group])

#: The only columns a model is allowed to see.
MODEL_FEATURES: tuple[str, ...] = features.FEATURES + PLAYER_INPUTS

#: The team-level columns on their own: the model before player detail was added.
TEAM_FEATURES: tuple[str, ...] = features.FEATURES

#: The closing market chance of a *finished* game. It is never a model input.
#: It is a second answer key, next to who won, that part of the model learns
#: from on past seasons. See ``model.py``.
TEACHER = "teacher_home_prob"


def build_frame(tables: dict[str, pd.DataFrame], *, lineup_mode: str = "posted") -> pd.DataFrame:
    """One row per game with every model input and a few descriptive columns.

    ``lineup_mode`` is ``posted`` to use announced lineups where they exist
    (what is known a few hours before a game) or ``projected`` to always guess
    the lineup from the team's recent games (what is known that morning).
    """
    frame = features.build_features(tables["games"], tables["team_games"], tables["pitcher_games"])
    detail = players.build_player_features(tables["games"], tables, lineup_mode=lineup_mode)
    frame = frame.merge(detail, on="game_id", how="left", validate="one_to_one")
    odds = tables.get("odds")
    frame[TEACHER] = float("nan")
    if odds is not None and len(odds) and "market_home_prob" in odds.columns:
        closing = odds.drop_duplicates("game_id").set_index("game_id")["market_home_prob"]
        finished = frame["home_win"].notna()
        frame.loc[finished, TEACHER] = frame.loc[finished, "game_id"].map(closing).to_numpy()
    return frame


def model_matrix(frame: pd.DataFrame, columns: tuple[str, ...] = MODEL_FEATURES) -> pd.DataFrame:
    """The columns a model may see, and nothing else."""
    return frame.loc[:, list(columns)].astype(float)
