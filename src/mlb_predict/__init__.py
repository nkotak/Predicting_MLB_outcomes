"""MLB win probabilities, tested against real betting odds.

Stages, each runnable from the ``mlb`` command:

    fetch     download raw data (Retrosheet, MLB box scores and player detail, ESPN odds)
    build     turn raw downloads into clean tables and check them
    backtest  season-by-season test of the model, the market and the betting agent
    train     fit the model and the agent that are used for new predictions
    predict   win chances and paper-bet suggestions for one date
    grade     score logged predictions once the games are final
    learn     update the betting agent with newly graded games
    scorecard running track record of logged predictions
    page      write the HTML results page

Paper trading only. Nothing in this package places a real bet.
"""

__version__ = "1.1.0"
