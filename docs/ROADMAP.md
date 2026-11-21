# Where this model stands, and the path to state of the art

Written after the player-level rebuild. Every number here comes from this
project's own season-by-season tests (`reports/BACKTEST.md`). No outside
benchmark is quoted, because none could be checked.

## 1. Where it stands

The scoring measure is log loss. Lower is better, and a coin flip scores
0.6931. The closing betting price is the best public forecast of a baseball
game that exists, so it is the yardstick.

| Forecaster | Correct | Log loss |
|---|---:|---:|
| Always the home team | 53.3% | 0.6910 |
| Team rating (Elo) alone | 56.6% | 0.6787 |
| This project before the player-level rebuild | 57.2% | 0.6765 |
| **This project now** | **57.6%** | **0.6746** |
| Now, on the 27,162 games that have odds | 57.7% | 0.6742 |
| Betting market's closing price, same games | 58.2% | 0.6720 |

The gap to the closing price was 0.0042 before this round and is 0.0022 now.

Read as "how much of what the market knows does the model know": going from
a coin flip to the closing price is the whole journey. The team-only model
covered about 80% of it. The current model covers about 90%.

What that means in practice:

- **Against the closing price, the model has nothing to add.** Mixing it
  with the closing price is no better than the closing price alone.
- **Against the opening price, it does.** Mixing it with the opening price
  beats the opening price alone by more than chance, even with lineups
  guessed that morning.
- **The price you get matters more than the model.** On games quoted by
  three or more sportsbooks, taking the best price instead of the middle
  price cuts the cost of a blind bet from about 3.5% to about 1.2%, and the
  loss on the model's picks from about 2.5% to about 1%, which cannot be
  told apart from break-even.

## 2. What was tried in this round

| Idea | Result |
|---|---|
| The nine hitters actually starting, each projected on his own | Kept. Helps |
| Starter's pitch quality: swings and misses, speed, ground balls | Kept. One of the two most useful additions |
| Age of hitters and starter | Kept. The most clearly useful addition, which says the projections should handle age themselves |
| Statcast contact quality, expected hitting, fielding range, speed | Kept. Helps |
| Each hitter's history against the starter | Kept. Tiny effect |
| Players on the injured list | Kept. Tiny effect once the real lineup is known, because the lineup already shows who is missing |
| Team fielding | Kept. Tiny effect |
| Expected runs from the full matchup (left/right, history, bullpen, ballpark) | Kept. The best single summary |
| Learning from run margins, not only wins | Kept. Small, steady gain |
| Learning from past closing prices | Kept. Gain larger than chance |
| Left/right matchups as separate inputs | Left out. Already inside expected runs |
| Bullpen, reliever by reliever, with tiredness | Left out. Made the forecasts slightly worse as separate inputs |
| Hot and cold streaks over three weeks | Left out. No value |
| Ballpark, weather, umpire as separate inputs | Left out. They change how many runs are scored, not who wins |
| Correcting player numbers for ballpark | Kept as a correction. No measurable change to the score |
| Deeper trees, more trees, a learned mix of the parts, weighting recent seasons more | No gain |

## 3. What the best models do that this one does not

In rough order of what I expect each to be worth. These are judgements, not
measurements.

1. **Pitch-by-pitch tracking data.** Statcast records the speed, spin,
   movement and location of every pitch. Models built on it ("stuff"
   models) can tell that a pitcher has changed within a few starts, long
   before his results show it. The tests here already point that way: the
   crude pitch-quality numbers in this project (swings and misses, fastball
   speed) were among the most useful things added. The data is free from
   Baseball Savant, about 700,000 pitches a season.
2. **Projections that are fitted, not set by hand.** The amounts by which
   each player is pulled toward average (220 plate appearances for hitters,
   350 for pitchers) and the weights on older seasons came from baseball
   common sense. They should be fitted to predict each player's next games,
   each skill separately (strikeouts settle in weeks, hits on balls in play
   take years), with age built in.
3. **A game simulation.** This model boils a game down to one expected
   run total per team. The best models play the game out many times, plate
   appearance by plate appearance: who bats when, when the starter comes
   out, which reliever comes in. That is the right way to use the
   reliever-by-reliever information that added nothing here as plain inputs,
   and it handles openers, bulk pitchers and extra innings properly.
4. **Minor league records** for players with little major league history.
   About one start in eleven is made by a pitcher with almost no record,
   whom this model treats as average. The same MLB feed carries minor league
   games. One caution: in the tests the model was not further from the
   market in those games than in others, so this may be worth less than it
   sounds.
5. **Weather forecasts and catcher framing.** Small.

## 4. What would help betting more than a better model

1. **More sportsbooks' prices.** The free feed has carried a single
   sportsbook since 2024. Shopping across books was worth about two points
   per bet in the seasons where it could be measured. No change to the model
   in this round moved the betting results by anything like that much.
2. **Bet early, and measure it honestly.** The model's information is in the
   opening price window. Run `mlb update` and `mlb predict` each morning in
   season. Every run logs the time and the price, and `mlb scorecard` then
   reports "value against the close": whether the prices taken beat the
   closing price. A few hundred logged bets will answer that. Profit and
   loss takes thousands.
3. **Use the mix, not the model alone.** For early bets, start from the
   opening price and move it toward the model by the tested amount. Bet only
   where that mix still disagrees with the price on offer by more than the
   bookmaker's cut. Teaching the agent to act on the mix is the natural next
   step for the GRPO work.
4. **Softer markets.** The main win market is the sharpest price in
   baseball. The same machinery here already produces expected runs for each
   team, which is what totals (over/under) are priced on, and the
   starter-against-lineup matchup is what first-five-innings bets are priced
   on. Both are usually softer.

## 5. An honest ceiling

Baseball is the most random of the major sports. The best team wins about
six games in ten. The closing price scores about 0.672 and nothing free and
public is known to beat it consistently. A realistic goal for this project
is not to beat the closing price. It is to be right about where the price
will end up, earlier than the market gets there, and to take the best price
available when it does. The results above say the first half of that is
within reach and the second half needs more sportsbooks' prices than the
free feed provides.
