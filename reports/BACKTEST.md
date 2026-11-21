# Backtest results

Generated 2026-11-21 by `mlb backtest`. Do not edit by hand.

**27,619 games**, seasons 2015 to 2026. Every season was predicted by a model that was trained and calibrated only on earlier seasons.

## How good are the forecasts?

"Correct" is the share of games where the side given the higher chance won. "Log loss" scores the probabilities themselves and lower is better. A coin flip scores 0.6931.

| Forecaster | Games | Correct | Log loss | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|
| Model | 27,619 | 57.2% | 0.6765 | 0.2418 | 0.46% |
| Logistic regression part only | 27,619 | 57.2% | 0.6764 | 0.2418 | 0.69% |
| Boosted trees part only | 27,619 | 57.3% | 0.6777 | 0.2424 | 0.61% |
| Team rating (Elo) only | 27,619 | 56.6% | 0.6787 | 0.2429 | 0.40% |
| Always pick the home team | 27,619 | 53.3% | 0.6910 | 0.2489 | 0.02% |

Model compared with always picking the home team: accuracy +3.9 points (95% range +3.3 to +4.6), log loss -0.0144 (95% range -0.0165 to -0.0125).

## Against the betting market

Closing prices are available for 27,162 of the tested games (98.3%). On those games:

| Forecaster | Correct | Log loss | Calibration error |
|---|---:|---:|---:|
| Betting market (closing price, bookmaker's cut removed) | 58.2% | 0.6720 | 0.12% |
| Model | 57.3% | 0.6762 | 0.41% |

Model log loss minus market log loss: +0.0042 (95% range +0.0032 to +0.0052). A value above zero means the market is the better forecaster.

## Season by season

| Season | Games | Always home | Model | Market | Model log loss | Market log loss |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 2,429 | 54.1% | 54.7% | 57.2% | 0.6838 | 0.6766 |
| 2016 | 2,427 | 53.0% | 56.9% | 58.5% | 0.6805 | 0.6730 |
| 2017 | 2,430 | 54.0% | 56.8% | 57.4% | 0.6784 | 0.6732 |
| 2018 | 2,429 | 52.8% | 58.0% | 58.5% | 0.6734 | 0.6668 |
| 2019 | 2,429 | 52.9% | 60.0% | 60.6% | 0.6656 | 0.6629 |
| 2020 | 898 | 55.7% | 57.8% | 60.3% | 0.6768 | 0.6715 |
| 2021 | 2,429 | 53.9% | 58.0% | 58.9% | 0.6741 | 0.6702 |
| 2022 | 2,430 | 53.3% | 59.9% | 60.5% | 0.6685 | 0.6646 |
| 2023 | 2,430 | 52.1% | 56.9% | 57.5% | 0.6780 | 0.6759 |
| 2024 | 2,429 | 52.2% | 56.9% | 57.4% | 0.6784 | 0.6741 |
| 2025 | 2,430 | 54.3% | 55.5% | 55.5% | 0.6793 | 0.6760 |
| 2026 | 2,429 | 52.9% | 55.7% | 57.1% | 0.6818 | 0.6791 |

## Paper betting at closing prices

Seasons 2018 to 2026, 20,059 games with odds. No real money was involved. This is the hardest test, because the closing price already contains everything the market learned that day. "Return per unit" is the profit per 1 unit staked if every bet were the same size, with a 95% range from resampling the bets. "Bankroll" starts at 1,000 units and uses each strategy's own stake sizes.

| Strategy | Bets | Won | Avg price | Return per unit | 95% range | Final bankroll | Worst drop |
|---|---:|---:|---:|---:|---|---:|---:|
| GRPO agent | 0 | | | no bets | | 1,000 | |
| Edge rule (3%+, quarter Kelly) | 8,778 | 42.9% | 2.35 | -3.6% | -5.9% to -1.1% | 2 | 99.8% |
| Always the home team | 20,059 | 53.2% | 1.88 | -3.6% | -4.9% to -2.3% | 0 | 100.0% |
| Always the favourite | 20,059 | 58.4% | 1.67 | -3.5% | -4.7% to -2.4% | 0 | 100.0% |
| Always the underdog | 20,059 | 41.6% | 2.37 | -3.7% | -5.3% to -2.0% | 0 | 100.0% |

### The agent, season by season

"Kept never-bet" means that after training, no version of the agent was proven better than not betting on the held-back season, so it passed on every game.

| Season | Trained on | Checked on | Result of training | Bets | Return per unit | Edge rule return per unit |
|---|---|---|---|---:|---:|---:|
| 2018 | 2015 to 2016 | 2017 | kept never-bet | 0 | no bets | -4.9% |
| 2019 | 2015 to 2017 | 2018 | kept never-bet | 0 | no bets | 1.0% |
| 2020 | 2015 to 2018 | 2019 | kept never-bet | 0 | no bets | -3.4% |
| 2021 | 2015 to 2019 | 2020 | kept never-bet | 0 | no bets | -1.0% |
| 2022 | 2015 to 2020 | 2021 | kept never-bet | 0 | no bets | -7.1% |
| 2023 | 2015 to 2021 | 2022 | kept never-bet | 0 | no bets | -0.8% |
| 2024 | 2015 to 2022 | 2023 | kept never-bet | 0 | no bets | -5.6% |
| 2025 | 2015 to 2023 | 2024 | kept never-bet | 0 | no bets | -6.7% |
| 2026 | 2015 to 2024 | 2025 | kept never-bet | 0 | no bets | -5.0% |

What the learning looked like: averaged over these seasons, the agent started out betting on 63% of its training games and ended up betting on 12%. On the held-back season, what it still wanted to bet on did not make money.

## Betting early, at opening prices

Opening prices are available for 7,240 tested games (seasons 2024 to 2026).

- **The price moves toward the model.** For each point the model sat away from the opening price, the price moved 0.39 of a point in the model's direction by game time. The link between the two is 0.50 on a scale where 0 means none (anything inside plus or minus 0.02 would be noise).
- **Opening prices are softer than closing prices.** Log loss: opening price 0.6790, closing price 0.6765, model 0.6800. The price moved 2.4 points on average between open and close.

One caution. The model is given the pitcher who actually started. When an opening price was set before a late change of starter, the model knew something that nobody could have known at the time. That flatters these numbers by an unknown amount. The prediction log, which records the price at the moment a prediction is made, is the fair way to measure this from now on.

Paper bets at the opening price, 2,429 games (season 2026). "Value against close" compares the price taken with the fair closing price. Above zero means the bets beat the closing line, which is the usual early sign of a real edge.

| Strategy | Bets | Won | Avg price | Return per unit | 95% range | Value against close |
|---|---:|---:|---:|---:|---|---:|
| GRPO agent | 0 | | | no bets | | |
| Edge rule (3%+, quarter Kelly) | 775 | 45.7% | 2.20 | -1.9% | -9.6% to 5.7% | -1.99% |
| Always the home team | 2,429 | 52.9% | 1.86 | -4.0% | -7.5% to -0.3% | -3.67% |
| Always the favourite | 2,429 | 56.0% | 1.71 | -5.1% | -8.5% to -1.8% | -4.27% |
| Always the underdog | 2,429 | 44.0% | 2.22 | -3.7% | -7.9% to 0.6% | -4.62% |

| Season | Trained on | Checked on | Result of training | Bets | Return per unit | Edge rule return per unit |
|---|---|---|---|---:|---:|---:|
| 2026 | 2024 | 2025 | kept never-bet | 0 | no bets | -1.9% |

## Inputs the model is allowed to see

`d_win_pct`, `d_pythag`, `d_rs_pg`, `d_ra_pg`, `d_ops`, `d_iso`, `d_bat_k`, `d_bat_bb`, `d_form10`, `d_rd30`, `d_pen_ra9`, `d_pen_kbb`, `d_pen_outs3`, `d_rest`, `d_elo`, `d_sp_k`, `d_sp_bb`, `d_sp_hr`, `d_sp_ra9`, `d_sp_fip`, `d_sp_whip`, `d_sp_outs_per_start`, `d_sp_gs5`, `d_sp_exp`, `d_sp_rest`, `home_sp_known`, `away_sp_known`, `season_progress`, `is_doubleheader`, `is_night`

---
The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.
