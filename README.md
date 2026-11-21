# MLB Predict

Win chances for every Major League Baseball game, tested against real betting
prices, with a learning agent that makes paper-bet decisions.

**Paper trading only. Nothing in this project places a real bet.**

<!-- results:start (written by `mlb backtest`, do not edit by hand) -->
## What the results say

Tested on **27,619 games** from seasons 2015 to 2026. Every season was predicted by a model that had never seen it.

| Forecaster | Games | Called correctly |
|---|---:|---:|
| This model | 27,619 | **57.2%** |
| Team rating (Elo) on its own | 27,619 | 56.6% |
| Always picking the home team | 27,619 | 53.3% |
| Betting market's closing price | 27,162 | **58.2%** |
| This model, on those same games | 27,162 | 57.3% |

In plain words:

- **The model is real.** It beats always picking the home team by 3.9 points, and it cannot see a game's own result.
- **It does not beat the betting market.** The closing price is the better forecaster (log loss 0.6720 against 0.6762, and the gap is larger than chance).
- **Paper betting at closing prices has not made money** (seasons 2018 to 2026, 20,059 games). Blind strategies lost 3.5% to 3.7% per unit staked, which is the bookmaker's cut. A fixed "bet when the model sees a 3% edge" rule returned -3.6%.
- **The learning agent chose not to bet.** In all 9 test seasons, no betting policy it learned was proven better than not betting, so it passed on every game. That is the right call given the line above.
- **What the agent would have bet with its safety bar off:** 2,339 paper bets, returning -4.5% per unit staked. That simulation is why the bar exists.
- **One promising signal: early prices.** When the model disagrees with the opening price, the price tends to move toward the model by game time (7,240 games, seasons 2024 to 2026). That has not yet turned into profit after the bookmaker's cut. The prediction log is there to test it fairly from now on.

Full numbers, season by season: [reports/BACKTEST.md](reports/BACKTEST.md). The same results as a page: `reports/index.html`.
<!-- results:end -->

## What you get

- `mlb predict --date 2027-04-02` prints each game's home win chance, the
  market's view, the edge on each side, and what the agent would do with a
  paper bankroll.
- Every prediction is logged with the time and the price at that moment.
  `mlb grade` scores it after the game, and `mlb scorecard` shows the running
  record, including whether your prices beat the closing price.
- `mlb backtest` re-runs the full season-by-season test and rewrites the
  reports. `mlb page` rebuilds the results page at `reports/index.html`.

## Quick start

Needs Python 3.10 or newer.

```bash
pip install -e ".[dev]"
```

```bash
mlb fetch
```

```bash
mlb build
```

```bash
mlb backtest
```

```bash
mlb train
```

```bash
mlb predict --date 2027-03-25
```

`mlb fetch` downloads everything. Game history takes a few minutes. Betting
odds take one to two hours the first time, because they come one game at a
time. The download can be stopped and restarted, and `mlb fetch --no-odds`
skips the odds if you only want forecasts.

Data is saved under `data/` and models under `models/`. Neither is committed.
Set `MLB_DATA_DIR` or `MLB_MODELS_DIR` to keep them somewhere else.

## A day in season

```bash
mlb update
```

```bash
mlb grade
```

```bash
mlb predict
```

- `mlb update` pulls new results, announced starters and current prices, then
  rebuilds and re-checks the data. It takes about a minute.
- `mlb grade` scores yesterday's logged predictions.
- `mlb predict` shows today's games. Run it early in the day: the evidence in
  this project says early prices are softer than closing prices.

Once a week:

```bash
mlb learn
```

```bash
mlb scorecard
```

`mlb learn` lets the agent keep learning from the newest results, using the
prices you actually saw when you logged predictions.

## How it works

Plain-language detail is in [docs/METHODS.md](docs/METHODS.md). Every table
and column is described in [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

| Step | What happens | Code |
|---|---|---|
| Data | Retrosheet for finished seasons, MLB's public feed for the current season and upcoming games, ESPN's public feed for prices. Players and teams are matched by id. The build stops if a data check fails | `sources/`, `build.py` |
| Inputs | 30 numbers per game about team strength, the two starting pitchers and context, each built only from earlier games | `features.py` |
| Model | Logistic regression plus boosted trees, then calibrated on a held-back season | `model.py` |
| Scoring | Each season is predicted by a model that never saw it, and compared with simple yardsticks and the market | `evaluate.py`, `backtest.py` |
| Agent | A GRPO policy that learns when to bet and how much from past results. It passes unless betting is proven better on held-back games | `agent.py` |
| Betting maths | One definition of edge, day-by-day paper settlement, value against the closing price | `betting.py` |
| Daily use | Predictions, the prediction log, grading, scorecard, continued learning | `predict.py`, `cli.py` |

### The no-peeking rule

An earlier version of this project reported 91% accuracy because it used each
game's own box score as an input. That cannot happen here:

- every input is built from earlier games only
- the model reads an explicit list of allowed columns and nothing else
- a test rewrites one game's result and checks that its own inputs do not move

### About the GRPO agent

GRPO (Group Relative Policy Optimization) is a trial-and-reward learning
method. For each past game the agent tries a group of actions (pass, or back
either side with a small, medium or large stake), sees what each would have
returned, and shifts toward the ones that did better than the rest of the
group. The policy is a small neural network. Its maths is checked in the
tests, and two tests prove it learns: it finds a betting edge that is planted
in made-up data, and it learns to stop betting when the market is right.

The agent only bets when a trained policy beats "never bet" on a season it
did not train on. If nothing clears that bar it passes on every game.

## Tests

```bash
pytest
```

57 tests, no network needed. They run automatically on every push, on the
oldest supported library versions (Python 3.10, pandas 2.2) and the newest.

## What this project cannot do

- It does not know line-ups, injuries or weather. The betting market does.
- It has not shown a betting profit. See the results above before trusting it
  with anything.
- Betting prices come from an unofficial ESPN feed that could change or stop.
- Early-price results use the pitcher who actually started, which a bettor
  that morning might not have known. The prediction log is the fair measure.

## Repository layout

```
src/mlb_predict/   the package
tests/             the test suite
docs/              methods and data dictionary
reports/           backtest results and the results page (generated)
legacy/            the original 2020 notebooks, archived and not maintained
```

## Credits and data terms

The information used here was obtained free of charge from and is copyrighted
by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.

Player ids are linked using the Chadwick Bureau register. Current-season
results and schedules come from MLB's public feed, and betting prices from
ESPN's public feed. Both are used for personal, non-commercial analysis.
Check their terms before publishing or selling anything built on them. Raw
data from these sources is not stored in this repository.

The page image in `reports/assets/` was generated with Higgsfield.

This project started as a 2020 student capstone. The original notebooks are
kept in `legacy/`.
