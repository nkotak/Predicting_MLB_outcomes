# Data dictionary

Every table and column the project uses, where it comes from, and when the
value becomes known relative to the game. That last point is what keeps the
model honest: **nothing that is only known after a game starts may be used to
predict that game.**

All tables live in `data/processed/` as Parquet files. They are rebuilt by
`mlb build` and are never committed to the repository.

## Where the data comes from

| Source | What it supplies | Seasons | Ids used |
|---|---|---|---|
| Retrosheet season files | Finished games: one row per game, per team, per pitcher appearance | 2010 to the latest season Retrosheet has published | Retrosheet player ids, converted to MLB ids during the build |
| MLB public schedule and box score feed | The season in progress, any season Retrosheet has not published yet, and upcoming games with announced starters | Current and next season | MLB player and team ids |
| ESPN public odds feed | Moneyline prices from several sportsbooks per game | 2014 onward. Opening prices for recent seasons only | ESPN event and team ids |
| Chadwick Bureau register | The link between a player's Retrosheet id and MLB id | All | Both |

Team codes from all three sources go through one table in
`src/mlb_predict/teams.py`. An unknown code stops the build.

## `games`: one row per game

| Column | Meaning | Known before the game? |
|---|---|---|
| `game_id` | `YYYYMMDD-HOME-N`. N is 0 for a single game, 1 or 2 for a doubleheader | yes |
| `date`, `season` | Official game date and season | yes |
| `number` | 0, 1 or 2 as above | yes |
| `home`, `away` | Team codes | yes |
| `home_sp_id`, `away_sp_id` | Starting pitcher (MLB id). For a finished game, the pitcher who actually started. For an upcoming game, the announced starter, or empty if not announced | yes for upcoming games. See the caution below |
| `day_night`, `park` | Day or night, ballpark | yes |
| `status` | `final` or `scheduled` | yes |
| `home_score`, `away_score` | Final score | **no** |
| `source` | `retrosheet` or `mlbapi` | |
| `mlb_game_pk` | MLB's game number, when the row came from MLB | |

Caution on starters: for finished games the table holds the real starter. If a
starter was changed shortly before a game, a bettor that morning would have
seen a different name. This matters when testing against opening prices.

## `team_games`: two rows per finished game

Each team's totals for that one game. **None of these are known before the
game.** They are only ever used for later games.

`pa`, `ab`, `runs`, `h`, `doubles`, `triples`, `hr`, `bb`, `ibb`, `hbp`, `so`,
`sf`, `sb`, `cs`, `lob` are the team's batting totals. `p_outs`, `p_bf`, `p_h`,
`p_hr`, `p_r`, `p_er`, `p_bb`, `p_so`, `p_hbp` are its pitching totals.
`errors` is fielding errors. `is_home` marks the home side.

## `pitcher_games`: one row per pitcher appearance

A pitcher's own line for one game: `outs`, `bf` (batters faced), `h`, `hr`,
`r`, `er`, `bb`, `ibb`, `so`, `hbp`, plus `seq` (order of appearance) and
`is_start`. **Not known before the game.** Used only for later games.

## `players`

`player_id` (MLB id as text), `name`, `throws`.

## `odds`: one row per game that has a price

| Column | Meaning |
|---|---|
| `home_dec`, `away_dec` | Decimal price for each side: the middle price across sportsbooks. For a finished game this is the closing price. For an upcoming game it is the current price |
| `market_home_prob` | The home win chance those prices imply, with the bookmaker's cut removed |
| `n_books` | How many sportsbooks went into the middle price |
| `open_home_dec`, `open_away_dec`, `open_market_home_prob` | The same for the opening price, where the feed records one |
| `event_id`, `odds_fetched_utc` | ESPN's event id and when the price was downloaded |

Prices recorded while a game was in progress ("Live Odds" entries) are never
read. The build fails if a season's favourites win so often that the prices
must have been taken during games.

## Model inputs

The model sees exactly the 30 columns below and nothing else. They are listed
in `FEATURES` in `src/mlb_predict/features.py`. Every one is built from games
played **before** the game in question. Columns starting with `d_` are the
home team's value minus the away team's value.

### Team strength

| Input | Meaning |
|---|---|
| `d_win_pct` | Share of games won this season, blended with last season |
| `d_pythag` | Expected win rate from runs scored and allowed |
| `d_rs_pg`, `d_ra_pg` | Runs scored and allowed per game |
| `d_ops`, `d_iso` | Batting: on-base plus slugging, and extra-base power |
| `d_bat_k`, `d_bat_bb` | Batting strikeout and walk rates |
| `d_form10` | Win rate over the last 10 games this season, home and road games together |
| `d_rd30` | Run difference per game over the last 30 games |
| `d_pen_ra9`, `d_pen_kbb` | Bullpen runs allowed per nine innings, and strikeout rate minus walk rate |
| `d_pen_outs3` | Outs the bullpen recorded over the previous three days (tiredness) |
| `d_rest` | Days since the team's last game |
| `d_elo` | A running team rating that rises with wins and falls with losses |

"Blended with last season" means a team starts each season with the weight of
about 40 games of last year's record (pulled halfway to the league average),
and its own results take over as the season goes on.

### Starting pitcher

| Input | Meaning |
|---|---|
| `d_sp_k`, `d_sp_bb`, `d_sp_hr` | His strikeout, walk and home run rates per batter faced |
| `d_sp_ra9` | Runs he allowed per nine innings |
| `d_sp_fip` | A run estimate built only from strikeouts, walks and home runs |
| `d_sp_whip` | Hits and walks per inning |
| `d_sp_outs_per_start` | How deep he usually pitches |
| `d_sp_gs5` | Quality of his last five starts on a 0 to 100 style scale |
| `d_sp_exp` | How much history he has (more history, more trust) |
| `d_sp_rest` | Days since he last pitched |

These come from the pitcher's own lines, this season plus earlier seasons
counted for less (last year 0.6, two years ago 0.35, older 0.2). A pitcher
with little history is pulled toward the league average.

### Context

`home_sp_known`, `away_sp_known` (0 when a starter has no history or is not
announced), `season_progress` (0 at opening day, 1 at the end),
`is_doubleheader`, `is_night`.

## The prediction log

`data/predictions/prediction_log.csv`, written by `mlb predict` and filled in
by `mlb grade`.

| Column | Meaning |
|---|---|
| `logged_utc` | When the prediction was made |
| `made_before_game` | True if the game had not been played when it was logged. Only these count as a real track record |
| `p_home` | Model's home win chance |
| `home_dec`, `away_dec`, `market_home_prob` | Prices at the moment of the prediction |
| `agent_side`, `agent_stake` | The GRPO agent's paper decision and stake (share of bankroll) |
| `rule_side`, `rule_stake` | The fixed edge rule's paper decision |
| `graded`, `home_win` | Filled in after the game |
| `agent_profit`, `rule_profit` | Profit per unit staked |
| `close_home_dec`, `close_away_dec` | Closing prices, used to check whether a bet beat the close |
