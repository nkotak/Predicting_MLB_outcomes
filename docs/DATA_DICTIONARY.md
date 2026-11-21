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
| MLB public game feed | For every game: both starting lineups, every plate appearance (batter, pitcher, which side each stands on, what happened, a summary of the pitches), the active roster, the weather and the home plate umpire | 2010 onward, one download per game | MLB player and team ids |
| MLB public transactions feed | Injured list moves (called the disabled list before 2019) | 2010 onward | MLB player and team ids |
| Baseball Savant season tables (Statcast) | Expected hitting from quality of contact, fielders' outs above average, sprint speed | 2015 onward | MLB player ids |
| ESPN public odds feed | Moneyline prices from several sportsbooks per game | 2014 onward. Opening prices for recent seasons only | ESPN event and team ids |
| Chadwick Bureau register | The link between a player's Retrosheet id and MLB id | All | Both |

Team codes from all sources go through one table in
`src/mlb_predict/teams.py`. An unknown code stops the build.

Old seasons come from two places at once: Retrosheet for the box scores and
MLB's feed for the player detail. The two are joined on MLB's game number,
which is matched to each Retrosheet game by date, teams and game number. The
build checks that both name the same starting pitcher in at least 99% of
games (in practice they agree every time).

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

## `lineups`: nine rows per team per game

The starting lineup. `slot` is the batting order position (1 to 9),
`player_id` the hitter and `position` where he started in the field. For a
finished game this is the lineup that actually started. For an upcoming game
it is the posted lineup, once the club has announced it.

Known before the game: **yes, but only a few hours before.** A forecast made
in the morning has to guess. `mlb predict` says which it used, and the
backtest reports both.

## `rosters`: about 26 rows per team per game

Everyone on the active roster that day. `role` is `SP` (that day's starter),
`RP` (a reliever who pitched), `BP` (a pitcher who did not), `BAT` (a hitter
who played) or `BN` (a hitter who did not). Who is on the roster is known
before the game. Who ended up playing is not, so the model only uses the
roster to answer "which relievers were available?" and "is this hitter still
with the club?".

## `plate_appearances`: one row per batter per pitcher per game

What each hitter did against each pitcher he faced in one game. **Not known
before the game.** Used only for later games.

| Column | Meaning |
|---|---|
| `batter`, `pitcher`, `bat_team`, `pit_team` | Who and for which clubs |
| `bat_side`, `pit_hand` | The side the hitter stood on and the hand the pitcher threw with (L or R) |
| `vs_starter` | True when the pitcher was his team's starter that day |
| `pa`, `ab`, `h1`, `h2`, `h3`, `hr`, `bb`, `ibb`, `hbp`, `so`, `sf` | Plate appearances, at bats, singles, doubles, triples, home runs, unintentional walks, intentional walks, hit by pitch, strikeouts, sacrifice flies |
| `woba_num`, `woba_den` | Top and bottom of weighted on-base average (wOBA): one number per hitter that counts a home run for more than a single |
| `pitches`, `whiffs`, `called`, `swings` | Pitches thrown, swings and misses, called strikes, swings |
| `fb_speed_sum`, `fb_n` | Total speed and count of fastballs, so that average fastball speed can be worked out over any stretch |
| `bip`, `bip_hit` | Balls put in play that a fielder could reach, and how many fell for hits |
| `tracked`, `gb` | Batted balls with a recorded flight, and how many were ground balls |
| `ev_n`, `ev_sum`, `hard` | Batted balls with a measured speed off the bat, the total of those speeds, and how many were hit 95 mph or harder. Measured from 2015 on |

## `game_extras`: one row per game

`venue` (MLB's ballpark number), `temp` (degrees F), `wind_mph`, `wind_out`
(the wind speed when it blows out to the outfield, minus when it blows in,
zero for a cross wind), `roof_closed`, `sky`, `umpire` (home plate). The
weather is the reading published with the game. For upcoming games it is
filled in when MLB posts it, and is empty until then.

## `people`

`player_id`, `name`, `bats` (L, R, or S for a switch hitter), `throws`,
`birth_date`, `position`.

## `il_spells`: one row per stretch on the injured list

`player_id`, `team`, `start`, `end`. A stretch starts on the day the move
was announced, so nothing in it was unknown at the time.

## `sc_expected`, `sc_fielding`, `sc_speed`: Statcast season tables

One row per player per season: `xwoba` next to actual `woba` (for hitters and
for pitchers), `oaa` (outs above average in the field) and `sprint_speed`.
These are whole-season numbers. **A game is only ever given the numbers from
the season before it.**

## `odds`: one row per game that has a price

| Column | Meaning |
|---|---|
| `home_dec`, `away_dec` | Decimal price for each side: the middle price across sportsbooks. For a finished game this is the closing price. For an upcoming game it is the current price |
| `market_home_prob` | The home win chance those prices imply, with the bookmaker's cut removed |
| `n_books` | How many sportsbooks went into the middle price |
| `best_home_dec`, `best_away_dec` | The best price any one of those sportsbooks offered on each side: what shopping around gets you |
| `open_home_dec`, `open_away_dec`, `open_market_home_prob` | The same for the opening price, where the feed records one |
| `event_id`, `odds_fetched_utc` | ESPN's event id and when the price was downloaded |

Prices recorded while a game was in progress ("Live Odds" entries) are never
read. The build fails if a season's favourites win so often that the prices
must have been taken during games.

## Model inputs

The model sees exactly the 58 columns below and nothing else. They are listed
in `MODEL_FEATURES` in `src/mlb_predict/inputs.py`: 30 team-level columns
built in `features.py` and 28 player-level columns built in `players.py`.
Every one is built from games played **before** the game in question, plus
what is announced before first pitch (the starters, the lineup, the roster).
Columns starting with `d_` are the home team's value minus the away team's
value.

"wOBA" below means weighted on-base average: one number for hitting that
counts a home run for more than a single. Around .320 is average.

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

### Lineup (player view)

| Input | Meaning |
|---|---|
| `d_lu_woba` | Projected wOBA of the nine hitters starting, weighted by how often each lineup slot bats |
| `d_lu_k`, `d_lu_bb`, `d_lu_iso` | The lineup's projected strikeout rate, walk rate and extra-base power |
| `d_lu_exp` | How much major league history the lineup has |
| `d_lu_vs_usual` | Today's lineup against the team's average lineup over its last 20 games. Below zero means regulars are resting or hurt |

Each hitter is projected from his own plate appearances: this season in
full, last season at 0.6, the one before at 0.35, older at 0.2, pulled toward
the league average with the weight of 220 plate appearances.

### History against the starter

| Input | Meaning |
|---|---|
| `d_hist` | How far the lineup's past results against this exact pitcher move its expected hitting. A small nudge, because the samples are tiny |
| `d_hist_pa` | How many times the lineup has faced him before |

### Starter's pitches

| Input | Meaning |
|---|---|
| `d_sp_woba` | wOBA he has allowed |
| `d_sp_whiff`, `d_sp_csw` | Swings and misses per pitch, and called strikes plus swings and misses per pitch |
| `d_sp_gb` | Share of batted balls against him that are ground balls |
| `d_sp_speed` | His average fastball speed against the league's that season |
| `d_sp_speed_trend` | His fastball speed over his last three outings against his own norm |

### Defense

`d_def_der`: share of balls in play the team's fielders turn into outs.

### Statcast

| Input | Meaning |
|---|---|
| `d_lu_hard`, `d_sp_hard` | Share of batted balls hit 95 mph or harder: by the lineup, and against the starter. Measured from 2015 |
| `d_lu_xgap`, `d_sp_xgap` | Last season's expected wOBA minus actual wOBA. Above zero means a hitter was unlucky last year (or a pitcher lucky) |
| `d_lu_oaa` | Last season's outs above average in the field for the nine starters, added up |
| `d_lu_speed` | Their average sprint speed last season |

### Injuries

| Input | Meaning |
|---|---|
| `d_il_bat`, `d_il_pit` | How much hitting and pitching each club has on the injured list. Each player counts for his share of a regular's playing time multiplied by how much better than a stand-in he has been |
| `d_il_n` | How many players each club has on the list |

### Age

`d_lu_age` (average age of the nine hitters) and `d_sp_age` (the starter's
age). Young players tend to beat their record and old players to fall short
of it.

### Expected runs

| Input | Meaning |
|---|---|
| `d_xr` | Runs the home team is expected to score minus the away team's. Built from each hitter against the starter (left/right matchup and past meetings included) for the share of the game the starter usually covers, and against the rested part of the bullpen for the rest, adjusted for the ballpark |
| `x_total` | The two teams' expected runs added together |

### The second answer key: past closing prices

The built frame also carries `teacher_home_prob`: the closing market chance
of each **finished** game. It is not a model input and is not in
`MODEL_FEATURES`. One part of the model is fitted to it on past seasons, the
same way the other parts are fitted to who won. It is empty for games that
have not been played, and the model is never shown the price of the game it
is predicting.

### Built, tested and left out

These columns are produced by `players.py` and appear in the built frame,
but the model is not given them, because adding them did not improve the
forecasts on seasons the model had not seen. The numbers are in
`reports/BACKTEST.md`.

| Group | Columns | Why it added nothing |
|---|---|---|
| Left/right matchups on their own | `d_lu_hand`, `d_sp_vs_lu`, `d_mu_woba`, `d_mu_k` | The same information is already inside expected runs |
| Bullpen, reliever by reliever | `d_bp_woba`, `d_bp_k`, `d_bp_tired` | The team view already has bullpen results and workload, and the available bullpen is inside expected runs |
| Hot and cold streaks | `d_lu_hot`, `d_sp_hot` | Three weeks of results say almost nothing that the longer record does not |
| Ballpark, weather, umpire | `park_rf`, `temp`, `wind_out`, `ump_runs` | They change how many runs are scored, not which team scores more |

## The prediction log

`data/predictions/prediction_log.csv`, written by `mlb predict` and filled in
by `mlb grade`.

| Column | Meaning |
|---|---|
| `logged_utc` | When the prediction was made |
| `made_before_game` | True if the game had not been played when it was logged. Only these count as a real track record |
| `p_home` | Model's home win chance |
| `lineups_posted` | True if both starting lineups had been announced when the prediction was made. False means at least one was guessed from recent games |
| `best_home_dec`, `best_away_dec` | The best price across sportsbooks at that moment |
| `home_dec`, `away_dec`, `market_home_prob` | Prices at the moment of the prediction |
| `agent_side`, `agent_stake` | The GRPO agent's paper decision and stake (share of bankroll) |
| `rule_side`, `rule_stake` | The fixed edge rule's paper decision |
| `graded`, `home_win` | Filled in after the game |
| `agent_profit`, `rule_profit` | Profit per unit staked |
| `close_home_dec`, `close_away_dec` | Closing prices, used to check whether a bet beat the close |
