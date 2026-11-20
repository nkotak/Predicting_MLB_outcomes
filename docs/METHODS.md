# Methods

How the project works and why each choice was made. Written to be read by
someone who has not seen the code.

## 1. The question

For each MLB game: what is the chance the home team wins? And then: is that
chance different enough from the betting price to be worth a bet?

The second question is much harder than the first. A betting price already
reflects line-ups, injuries, weather and the money of professional bettors,
and the bookmaker keeps a cut of about 4%. Being right more often than a coin
flip is easy. Being more right than the price is not.

## 2. Data

Three free public sources, described in `DATA_DICTIONARY.md`.

- **Retrosheet** for finished seasons. Its per-game files include every
  pitcher's own line, so starter statistics are real and not team totals.
- **MLB's public feed** for the current season and upcoming games.
- **ESPN's public feed** for sportsbook prices from 2014 onward.

Players are matched across sources by id through the Chadwick register, never
by name. Teams go through one code table.

`mlb build` refuses to finish if any of these checks fail:

- a game appears twice, or a finished game does not have exactly two teams
- runs in the team table do not match the final score
- a team-game has more than one starting pitcher, or a team code is unknown
- a full season has an implausible number of games or teams
- more than 1% of starting pitchers have no MLB id
- a season's betting favourites win more than 61.5% of the time, which would
  mean prices recorded during games had got in

## 3. No peeking

The most important rule in the project. An earlier version scored 91% by
accidentally using each game's own box score.

- Every input for a game is built only from games played earlier (an earlier
  date, or game one of the same day's doubleheader).
- Finished and upcoming games go through the same code, so there is no
  separate "live" path that could behave differently.
- The model can only read the columns named in `FEATURES`. Adding a column
  somewhere else cannot leak into it.
- A test rewrites one game's result and box score and checks that the inputs
  for that game, and for every earlier game, do not change at all.

## 4. The model

Two simple models are averaged, then calibrated.

- **Logistic regression** on standardized inputs.
- **Gradient boosted trees** (shallow, heavily regularized).
- **Calibration**: a one-input logistic fit on a season neither model trained
  on, so that a stated 60% happens about 60% of the time.

Simple on purpose. The signal in baseball is small, and in testing the
logistic regression alone is about as good as the blend.

## 5. Scoring: walk forward by season

For a test season Y:

1. train on every season up to Y-2 (the first season in the data only warms
   up the running statistics and is never trained on)
2. calibrate on season Y-1
3. score on season Y

Then move forward one season and repeat. Every reported number comes from a
model that had never seen the season it was scored on.

Scores:

- **Correct**: share of games where the favoured side won.
- **Log loss** and **Brier score**: they score the probabilities themselves.
  Lower is better. A coin flip has a log loss of 0.6931.
- **Calibration error**: average gap between stated chance and what happened.

Each score is shown next to yardsticks: always picking the home team, the
team rating on its own, and the betting market where odds exist. Differences
come with a 95% range from resampling the games, because one season of about
2,430 games moves accuracy by roughly two points on luck alone.

## 6. The market as a forecaster

For each game the prices from all pre-game sportsbooks are converted to a
fair win chance (bookmaker's cut removed) and the middle value is used. A
book far from the others is dropped. For finished games this is the closing
price.

The model is then compared with the market on the same games. This is the
test that decides whether betting could ever pay.

## 7. Paper betting

No real money is involved anywhere in this project.

- **Edge** has one definition: `model chance x decimal price - 1`, the
  expected profit per unit staked at the price on offer.
- **Settlement is by day.** All of a day's stakes are sized from the bankroll
  at the start of that day and settled together, because games overlap.
- **Return per unit** is profit per 1 unit staked with every bet the same
  size. It is the fairest single number and comes with a 95% range.
- **Value against the close** compares the price taken with the fair closing
  price. Above zero means the bet beat the closing line. This shows a real
  edge long before profits can be told apart from luck.

Strategies compared: the GRPO agent, a fixed rule (bet when edge is at least
3%, quarter-Kelly stake, capped at 2% of bankroll), and three blind baselines
(always home, always favourite, always underdog) that show the cost of the
bookmaker's cut.

Betting is tested twice: at closing prices for every season with odds, and at
opening prices for the recent seasons that have them.

## 8. The GRPO agent

The agent decides, for each game, whether to pass or to back one side with a
small, medium or large paper stake (0.5%, 1% or 2% of bankroll). It sees the
model's chance and the market price. It is not given a betting rule.

**How it learns.** GRPO (Group Relative Policy Optimization) is a
trial-and-reward method:

1. For every training game, sample a *group* of actions from the current
   policy.
2. Score each action by the log of bankroll growth it would have produced.
3. Judge each action *relative to its group*: subtract the group's average
   reward. Better than average is encouraged, worse is discouraged. No
   separate critic model is needed. This is the core idea of GRPO.
4. Update the policy with a clipped objective, so no single update can change
   an action's chance by more than 20%, plus a penalty for drifting from the
   previous policy.

The policy is a small neural network written in plain NumPy. Its gradients
are checked against finite differences in the tests.

**One deliberate change.** Original GRPO also divides each group's rewards by
that group's own spread. Here the spread inside one game depends on who won,
and dividing by it would quietly reward backing favourites whatever the
price. Rewards are therefore centred per game but scaled by the spread of the
whole batch. Later GRPO work makes the same change for a related reason.

**Passing is the default.** Single game results are mostly luck, so a policy
can look good by chance. After each learning round the policy is scored on a
held-back season. It is kept only if it beats "never bet" there by at least
two standard errors. If no round does, the agent keeps the never-bet policy.
An agent that passes on everything is a real result, not a failure to run.

**It keeps learning.** `mlb learn` continues training from the current policy
on the newest seasons, using the prices actually logged at prediction time
where they exist.

**Does it really learn?** Tests plant a known edge in made-up data (the model
knows the truth, the market is noisy) and check the agent finds it and bets
the right side. A second test makes the market right and the model noisy and
checks the agent learns to stop betting.

## 9. Known limits

- **Starters and opening prices.** The data holds the pitcher who actually
  started. An opening price set before a late change could not have known
  that. Early-price results are flattered by an unknown amount. The
  prediction log, which stores the price at the moment of each prediction, is
  the fair measure going forward.
- **No line-ups, injuries or weather.** The market has them. The model does
  not.
- **Odds source.** ESPN's feed is unofficial and its coverage and sportsbooks
  change by season. Opening prices exist for recent seasons only.
- **Rule changes.** The 2023 pitch clock and shift limits changed the game.
  Older seasons may teach slightly stale lessons.
- **2020** was a 60 game season and is kept, with its smaller weight.
- **Retrosheet timing.** A season appears on Retrosheet the winter after it
  ends. Until then that season comes from MLB's feed.
