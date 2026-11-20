"""One table of team codes for every data source and season.

Each club has one stable code used everywhere inside this package. The three
sources name clubs differently and two clubs changed code during 2010 to 2026,
so every lookup goes through this table. An unknown code raises an error
instead of being passed through, which is what let mismatches go unnoticed in
the earlier version of this project.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    code: str  # stable code used inside this package
    mlb_id: int  # statsapi.mlb.com team id
    espn_id: int  # ESPN team id
    retro: tuple[str, ...]  # every Retrosheet code used for the club since 2010
    name: str


TEAMS: tuple[Team, ...] = (
    Team("LAA", 108, 3, ("ANA",), "Los Angeles Angels"),
    Team("ARI", 109, 29, ("ARI",), "Arizona Diamondbacks"),
    Team("ATL", 144, 15, ("ATL",), "Atlanta Braves"),
    Team("BAL", 110, 1, ("BAL",), "Baltimore Orioles"),
    Team("BOS", 111, 2, ("BOS",), "Boston Red Sox"),
    Team("CHC", 112, 16, ("CHN",), "Chicago Cubs"),
    Team("CWS", 145, 4, ("CHA",), "Chicago White Sox"),
    Team("CIN", 113, 17, ("CIN",), "Cincinnati Reds"),
    Team("CLE", 114, 5, ("CLE",), "Cleveland Guardians"),
    Team("COL", 115, 27, ("COL",), "Colorado Rockies"),
    Team("DET", 116, 6, ("DET",), "Detroit Tigers"),
    Team("HOU", 117, 18, ("HOU",), "Houston Astros"),
    Team("KC", 118, 7, ("KCA",), "Kansas City Royals"),
    Team("LAD", 119, 19, ("LAN",), "Los Angeles Dodgers"),
    Team("MIA", 146, 28, ("FLO", "MIA"), "Miami Marlins"),
    Team("MIL", 158, 8, ("MIL",), "Milwaukee Brewers"),
    Team("MIN", 142, 9, ("MIN",), "Minnesota Twins"),
    Team("NYM", 121, 21, ("NYN",), "New York Mets"),
    Team("NYY", 147, 10, ("NYA",), "New York Yankees"),
    Team("ATH", 133, 11, ("OAK", "ATH"), "Athletics"),
    Team("PHI", 143, 22, ("PHI",), "Philadelphia Phillies"),
    Team("PIT", 134, 23, ("PIT",), "Pittsburgh Pirates"),
    Team("SD", 135, 25, ("SDN",), "San Diego Padres"),
    Team("SEA", 136, 12, ("SEA",), "Seattle Mariners"),
    Team("SF", 137, 26, ("SFN",), "San Francisco Giants"),
    Team("STL", 138, 24, ("SLN",), "St. Louis Cardinals"),
    Team("TB", 139, 30, ("TBA",), "Tampa Bay Rays"),
    Team("TEX", 140, 13, ("TEX",), "Texas Rangers"),
    Team("TOR", 141, 14, ("TOR",), "Toronto Blue Jays"),
    Team("WSH", 120, 20, ("WAS",), "Washington Nationals"),
)

CODES: tuple[str, ...] = tuple(t.code for t in TEAMS)
_BY_CODE = {t.code: t for t in TEAMS}
_BY_MLB = {t.mlb_id: t.code for t in TEAMS}
_BY_ESPN = {t.espn_id: t.code for t in TEAMS}
_BY_RETRO = {r: t.code for t in TEAMS for r in t.retro}


class UnknownTeam(KeyError):
    """Raised when a source uses a team code this table does not know."""


def from_retro(code: str) -> str:
    try:
        return _BY_RETRO[code]
    except KeyError:
        raise UnknownTeam(f"Retrosheet team code {code!r} is not in the team table") from None


def from_mlb_id(team_id: int) -> str:
    try:
        return _BY_MLB[int(team_id)]
    except KeyError:
        raise UnknownTeam(f"MLB team id {team_id!r} is not in the team table") from None


def from_espn_id(team_id: int | str) -> str:
    try:
        return _BY_ESPN[int(team_id)]
    except KeyError:
        raise UnknownTeam(f"ESPN team id {team_id!r} is not in the team table") from None


def name(code: str) -> str:
    return _BY_CODE[code].name


def mlb_id(code: str) -> int:
    return _BY_CODE[code].mlb_id
