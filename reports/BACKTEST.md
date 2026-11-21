# Backtest results

Generated 2026-11-21 by `mlb backtest`. Do not edit by hand.

**27,619 games**, seasons 2015 to 2026. Every season was predicted by a model that was trained and calibrated only on earlier seasons.

## How good are the forecasts?

"Correct" is the share of games where the side given the higher chance won. "Log loss" scores the probabilities themselves and lower is better. A coin flip scores 0.6931.

| Forecaster | Games | Correct | Log loss | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|
| Model | 27,619 | 57.6% | 0.6746 | 0.2409 | 0.40% |
| Logistic regression part only | 27,619 | 57.6% | 0.6755 | 0.2414 | 1.27% |
| Boosted trees part only | 27,619 | 57.3% | 0.6765 | 0.2418 | 0.95% |
| Run margin part only | 27,619 | 57.5% | 0.6771 | 0.2420 | 2.59% |
| Team rating (Elo) only | 27,619 | 56.6% | 0.6787 | 0.2429 | 0.40% |
| Always pick the home team | 27,619 | 53.3% | 0.6910 | 0.2489 | 0.02% |
| Price-taught part only (2017 on) | 22,763 | 57.7% | 0.6735 | 0.2404 | 0.56% |
| Whole model, same seasons (2017 on) | 22,763 | 57.9% | 0.6737 | 0.2405 | 0.39% |

The price-taught part learns from what closing prices said about past games. It needs a few seasons of prices first, so it is in use from 2017 on and is scored on those seasons, next to the whole model on the same seasons. It is never shown the price of the game it is predicting.

Model compared with always picking the home team: accuracy +4.3 points (95% range +3.6 to +5.0), log loss -0.0164 (95% range -0.0186 to -0.0142).

## Against the betting market

Closing prices are available for 27,162 of the tested games (98.3%). On those games:

| Forecaster | Correct | Log loss | Calibration error |
|---|---:|---:|---:|
| Betting market (closing price, bookmaker's cut removed) | 58.2% | 0.6720 | 0.12% |
| Model | 57.7% | 0.6742 | 0.31% |

Model log loss minus market log loss: +0.0022 (95% range +0.0014 to +0.0030). A value above zero means the market is the better forecaster.

## What the player-by-player detail added

The team view describes each club as a whole and each starter from his box scores (30 inputs). The player view adds the nine hitters actually starting, left/right matchups, the bullpen and more (28 inputs).

| Version | Correct | Log loss | Calibration error |
|---|---:|---:|---:|
| Team view only | 57.3% | 0.6761 | 0.43% |
| Team view plus player detail (the model) | 57.6% | 0.6746 | 0.40% |
| The model with lineups guessed that morning | 57.5% | 0.6751 | 0.39% |

The player detail made the forecasts better, by more than chance: log loss changed by -0.0016 (95% range -0.0023 to -0.0008). Below zero is better.

### What each group of inputs is worth

Each group was taken out of the model (or put in, if it had been left out) and the whole season-by-season test was run again. "Helps by" is how much better the log loss is with the group than without it, in thousandths, with a 95% range. A range that includes zero means the effect cannot be told apart from chance.

| Group of inputs | In the model | Helps by (95% range) | Verdict |
|---|---|---:|---|
| Starter's pitches: swings and misses, speed, ground balls | yes | +0.30 (+0.00 to +0.61) | too small to tell from chance |
| The nine hitters actually starting | yes | +0.28 (-0.02 to +0.61) | too small to tell from chance |
| Age of the hitters and the starter | yes | +0.25 (+0.12 to +0.38) | helps |
| Statcast: contact quality, expected hitting, fielding range, speed | yes | +0.21 (+0.05 to +0.38) | helps |
| Left/right matchups, hitters against the starter | no, tested and left out | +0.05 (-0.05 to +0.14) | too small to tell from chance |
| Each hitter's past record against this starter | yes | +0.04 (-0.03 to +0.11) | too small to tell from chance |
| Expected runs, everything rolled together | yes | +0.04 (-0.18 to +0.25) | too small to tell from chance |
| Players on the injured list | yes | +0.02 (-0.07 to +0.11) | too small to tell from chance |
| Team fielding | yes | +0.01 (-0.06 to +0.08) | too small to tell from chance |
| Ballpark, weather and umpire | no, tested and left out | -0.01 (-0.12 to +0.10) | too small to tell from chance |
| Hot and cold streaks: the last three weeks against the longer record | no, tested and left out | -0.04 (-0.10 to +0.02) | too small to tell from chance |
| Bullpen quality and tiredness | no, tested and left out | -0.08 (-0.16 to +0.00) | hurts |

## Does the model know anything the market does not?

A simple mix of the market's chance and the model's chance was fitted on earlier seasons and scored on the next one. If the mix scores no better than the market on its own, the model holds nothing the price does not already contain. For opening prices the model uses lineups guessed that morning.

- Closing price, 22,415 games: market alone 0.6715, market mixed with the model 0.6715, change +0.0000 (95% range +0.0000 to +0.0001), so the model adds nothing to this price that can be told apart from chance. The mix gives the model 5% of the weight.
- Opening price, 4,858 games: market alone 0.6807, market mixed with the model 0.6795, change -0.0012 (95% range -0.0022 to -0.0001), so the model adds real information to this price. The mix gives the model 59% of the weight.

## Shopping for the best price

The betting tests use the middle price across sportsbooks. Someone with several accounts can take the best price on offer instead. This test uses the 13,165 games (seasons 2018 to 2024) that at least three sportsbooks quoted. From 2024 the free feed carries one sportsbook only, so there is nothing to shop between in those seasons.

At the best closing price:

| Strategy | Bets | Won | Avg price | Return per unit | 95% range | Final bankroll | Worst drop |
|---|---:|---:|---:|---:|---|---:|---:|
| GRPO agent | 0 | | | no bets | | 1,000 | |
| GRPO agent, safety bar off | 4,792 | 55.1% | 1.86 | -0.6% | -3.2% to 2.1% | 490 | 81.4% |
| Edge rule (3%+, quarter Kelly) | 7,336 | 44.6% | 2.33 | -1.0% | -3.6% to 1.6% | 88 | 93.2% |
| Always the home team | 13,165 | 53.1% | 1.94 | -1.4% | -3.0% to 0.3% | 78 | 92.9% |
| Always the favourite | 13,165 | 59.3% | 1.69 | -0.7% | -2.2% to 0.6% | 238 | 79.6% |
| Always the underdog | 13,165 | 40.7% | 2.49 | -1.4% | -3.5% to 0.7% | 59 | 96.7% |

The same games at the middle closing price:

| Strategy | Bets | Won | Avg price | Return per unit | 95% range | Final bankroll | Worst drop |
|---|---:|---:|---:|---:|---|---:|---:|
| GRPO agent | 34 | 44.1% | 2.63 | 11.9% | -29.5% to 56.6% | 1,042 | 5.0% |
| GRPO agent, safety bar off | 1,781 | 63.1% | 1.61 | -0.6% | -4.3% to 3.2% | 1,050 | 23.8% |
| Edge rule (3%+, quarter Kelly) | 4,735 | 44.1% | 2.32 | -2.5% | -5.7% to 0.9% | 161 | 88.2% |
| Always the home team | 13,165 | 53.1% | 1.89 | -3.6% | -5.2% to -1.9% | 5 | 99.6% |
| Always the favourite | 13,165 | 59.2% | 1.66 | -2.9% | -4.2% to -1.5% | 15 | 98.7% |
| Always the underdog | 13,165 | 40.8% | 2.42 | -3.9% | -5.9% to -1.9% | 2 | 99.8% |

On the 13,165 games that three or more sportsbooks quoted (seasons 2018 to 2024), betting blind costs about 3.5% per unit staked at the middle price and about 1.2% at the best price on offer. The model's own picks (the 3% edge rule) returned -2.5% at the middle price and -1.0% at the best price (95% range -3.6% to 1.6%).

## Season by season

| Season | Games | Always home | Model | Market | Model log loss | Market log loss |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 2,429 | 54.1% | 55.7% | 57.2% | 0.6801 | 0.6766 |
| 2016 | 2,427 | 53.0% | 56.8% | 58.5% | 0.6772 | 0.6730 |
| 2017 | 2,430 | 54.0% | 56.9% | 57.4% | 0.6773 | 0.6732 |
| 2018 | 2,429 | 52.8% | 58.6% | 58.5% | 0.6711 | 0.6668 |
| 2019 | 2,429 | 52.9% | 59.3% | 60.6% | 0.6637 | 0.6629 |
| 2020 | 898 | 55.7% | 57.7% | 60.3% | 0.6740 | 0.6715 |
| 2021 | 2,429 | 53.9% | 58.5% | 58.9% | 0.6726 | 0.6702 |
| 2022 | 2,430 | 53.3% | 60.8% | 60.5% | 0.6651 | 0.6646 |
| 2023 | 2,430 | 52.1% | 56.7% | 57.5% | 0.6792 | 0.6759 |
| 2024 | 2,429 | 52.2% | 57.3% | 57.4% | 0.6759 | 0.6741 |
| 2025 | 2,430 | 54.3% | 56.1% | 55.5% | 0.6786 | 0.6760 |
| 2026 | 2,429 | 52.9% | 56.8% | 57.1% | 0.6800 | 0.6791 |

## Paper betting at closing prices

Seasons 2018 to 2026, 20,059 games with odds. No real money was involved. This is the hardest test, because the closing price already contains everything the market learned that day. "Return per unit" is the profit per 1 unit staked if every bet were the same size, with a 95% range from resampling the bets. "Bankroll" starts at 1,000 units and uses each strategy's own stake sizes.

| Strategy | Bets | Won | Avg price | Return per unit | 95% range | Final bankroll | Worst drop |
|---|---:|---:|---:|---:|---|---:|---:|
| GRPO agent | 0 | | | no bets | | 1,000 | |
| GRPO agent, safety bar off | 1,283 | 64.4% | 1.57 | -1.3% | -5.5% to 2.6% | 967 | 24.2% |
| Edge rule (3%+, quarter Kelly) | 6,544 | 43.3% | 2.33 | -3.6% | -6.3% to -0.7% | 30 | 97.4% |
| Always the home team | 20,059 | 53.2% | 1.88 | -3.6% | -4.9% to -2.3% | 0 | 100.0% |
| Always the favourite | 20,059 | 58.4% | 1.67 | -3.5% | -4.7% to -2.4% | 0 | 100.0% |
| Always the underdog | 20,059 | 41.6% | 2.37 | -3.7% | -5.3% to -2.0% | 0 | 100.0% |

"Safety bar off" shows the bets the agent's learned policy would have placed if it were not required to prove itself first: 1,283 paper bets, 64.4% won, -1.3% return per unit staked (95% range -5.5% to 2.6%). The safety bar is what stopped those bets from being recommended.

### The agent, season by season

"Kept never-bet" means that after training, no version of the agent was proven better than not betting on the held-back season, so it passed on every game.

| Season | Trained on | Checked on | Result of training | Bets | Return per unit | Edge rule return per unit |
|---|---|---|---|---:|---:|---:|
| 2018 | 2015 to 2016 | 2017 | kept never-bet | 0 | no bets | -4.7% |
| 2019 | 2015 to 2017 | 2018 | kept never-bet | 0 | no bets | 1.2% |
| 2020 | 2015 to 2018 | 2019 | kept never-bet | 0 | no bets | -0.8% |
| 2021 | 2015 to 2019 | 2020 | kept never-bet | 0 | no bets | -0.3% |
| 2022 | 2015 to 2020 | 2021 | kept never-bet | 0 | no bets | -3.5% |
| 2023 | 2015 to 2021 | 2022 | kept never-bet | 0 | no bets | -8.0% |
| 2024 | 2015 to 2022 | 2023 | kept never-bet | 0 | no bets | -4.8% |
| 2025 | 2015 to 2023 | 2024 | kept never-bet | 0 | no bets | -9.8% |
| 2026 | 2015 to 2024 | 2025 | kept never-bet | 0 | no bets | -4.1% |

What the learning looked like: averaged over these seasons, the agent started out betting on 60% of its training games and ended up betting on 6%. On the held-back season, what it still wanted to bet on did not make money.

## Betting early, at opening prices

Opening prices are available for 7,240 tested games (seasons 2024 to 2026).

- **The price moves toward the model.** For each point the model sat away from the opening price, the price moved 0.54 of a point in the model's direction by game time. The link between the two is 0.61 on a scale where 0 means none (anything inside plus or minus 0.02 would be noise).
- **Opening prices are softer than closing prices.** Log loss: opening price 0.6790, closing price 0.6765, model 0.6784. The price moved 2.4 points on average between open and close.

For this section the model was not shown the real lineups. It used lineups guessed from each team's recent games, as a bettor would have to in the morning.

One caution. The model is still given the pitcher who actually started. When an opening price was set before a late change of starter, the model knew something that nobody could have known at the time. That flatters these numbers by an unknown amount. The prediction log, which records the price at the moment a prediction is made, is the fair way to measure this from now on.

Paper bets at the opening price, 2,429 games (season 2026). "Value against close" compares the price taken with the fair closing price. Above zero means the bets beat the closing line, which is the usual early sign of a real edge.

| Strategy | Bets | Won | Avg price | Return per unit | 95% range | Value against close |
|---|---:|---:|---:|---:|---|---:|
| GRPO agent | 0 | | | no bets | | |
| GRPO agent, safety bar off | 498 | 55.8% | 2.01 | 3.8% | -5.2% to 13.0% | -1.79% |
| Edge rule (3%+, quarter Kelly) | 636 | 48.3% | 2.13 | 0.6% | -7.6% to 8.6% | -0.37% |
| Always the home team | 2,429 | 52.9% | 1.86 | -4.0% | -7.5% to -0.3% | -3.67% |
| Always the favourite | 2,429 | 56.0% | 1.71 | -5.1% | -8.5% to -1.8% | -4.27% |
| Always the underdog | 2,429 | 44.0% | 2.22 | -3.7% | -7.9% to 0.6% | -4.62% |

| Season | Trained on | Checked on | Result of training | Bets | Return per unit | Edge rule return per unit |
|---|---|---|---|---:|---:|---:|
| 2026 | 2024 | 2025 | kept never-bet | 0 | no bets | 0.6% |

Paper bets on the model's picks at opening prices returned 0.6% per unit staked (636 bets, 95% range -7.6% to 8.6%). That range includes zero, so it could be luck either way.

## Inputs the model is allowed to see

`d_win_pct`, `d_pythag`, `d_rs_pg`, `d_ra_pg`, `d_ops`, `d_iso`, `d_bat_k`, `d_bat_bb`, `d_form10`, `d_rd30`, `d_pen_ra9`, `d_pen_kbb`, `d_pen_outs3`, `d_rest`, `d_elo`, `d_sp_k`, `d_sp_bb`, `d_sp_hr`, `d_sp_ra9`, `d_sp_fip`, `d_sp_whip`, `d_sp_outs_per_start`, `d_sp_gs5`, `d_sp_exp`, `d_sp_rest`, `home_sp_known`, `away_sp_known`, `season_progress`, `is_doubleheader`, `is_night`, `d_lu_woba`, `d_lu_k`, `d_lu_bb`, `d_lu_iso`, `d_lu_exp`, `d_lu_vs_usual`, `d_hist`, `d_hist_pa`, `d_sp_woba`, `d_sp_whiff`, `d_sp_csw`, `d_sp_gb`, `d_sp_speed`, `d_sp_speed_trend`, `d_def_der`, `d_lu_hard`, `d_sp_hard`, `d_lu_xgap`, `d_sp_xgap`, `d_lu_oaa`, `d_lu_speed`, `d_il_bat`, `d_il_pit`, `d_il_n`, `d_lu_age`, `d_sp_age`, `d_xr`, `x_total`

---
The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.
