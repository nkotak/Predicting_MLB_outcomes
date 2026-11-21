"""MLB Predict: win chances, tested against betting prices. Paper trading only.

Typical first run::

    mlb fetch        # download everything (takes two to three hours the first time)
    mlb build        # make and check the clean tables
    mlb backtest     # season-by-season test, written to reports/
    mlb train        # fit the model and the agent used for new predictions

Typical day in season::

    mlb update                     # new results, starters and prices
    mlb grade                      # score yesterday's logged predictions
    mlb predict --date 2027-04-02  # today's chances and paper bets
    mlb learn                      # once a week: let the agent learn from new results
    mlb scorecard                  # the track record so far
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import __version__

REPORTS = Path("reports")


def _date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date. Use YYYY-MM-DD, for example 2027-04-02.")


def _pct(value: float, digits: int = 1) -> str:
    return "n/a" if value != value else f"{100 * value:.{digits}f}%"


def cmd_fetch(args) -> int:
    from . import pipeline
    summary = pipeline.fetch_all(odds=not args.no_odds, refresh=args.refresh)
    print(f"Game history from Retrosheet: {summary['retrosheet_seasons'][0]} to {summary['retrosheet_seasons'][-1]}")
    print(f"Seasons from MLB's feed: {summary['mlb_seasons']}")
    print(f"Games with player detail downloaded now: {summary['game_feeds_downloaded']}")
    if summary["game_feeds_failed"]:
        print(f"Games that failed to download: {summary['game_feeds_failed']}. Run `mlb fetch` again to fill them in.")
    print(f"Odds requests made: {summary['odds_requests']}")
    print("Next: mlb build")
    return 0


def cmd_update(args) -> int:
    from . import pipeline
    summary = pipeline.update()
    print(f"New box scores: {summary['new_box_scores']}. New games with player detail: {summary['new_game_feeds']}. "
          f"Upcoming games checked for lineups: {summary['upcoming_games_checked']}. "
          f"Odds requests: {summary['odds_requests']}.")
    print(f"Finished games in data: {summary['build']['finished_games']}. Scheduled: {summary['build']['scheduled_games']}.")
    return 0


def cmd_build(args) -> int:
    from . import build
    report = build.run()
    print(f"Built {report['finished_games']} finished games and {report['scheduled_games']} scheduled games.")
    print(f"Starting pitchers matched by id: {_pct(report['starter_id_match_rate'], 2)}")
    if report.get("lineup_coverage"):
        low = min(report["lineup_coverage"].values())
        print(f"Finished games with both starting lineups: at least {_pct(low, 1)} in every season")
    covered = {season: share for season, share in report.get("odds_coverage", {}).items() if share > 0}
    if covered:
        print("Share of games with odds, by season: " + ", ".join(f"{s} {_pct(v, 0)}" for s, v in covered.items()))
    print("All data checks passed.")
    return 0


def cmd_backtest(args) -> int:
    from . import backtest, report
    result = backtest.run(save_to=REPORTS / "backtest.json", groups=args.groups)
    (REPORTS / "BACKTEST.md").write_text(report.backtest_markdown(result))
    print(report.backtest_text(result))
    print(f"\nFull results: {REPORTS / 'BACKTEST.md'} and {REPORTS / 'backtest.json'}")
    if report.update_readme(result):
        print("The results section of README.md was updated to match.")
    return 0


def cmd_train(args) -> int:
    from . import predict
    info = predict.train()
    model = info["model"]
    print(f"Model trained on seasons {model['train_seasons'][0]} to {model['train_seasons'][-1]} "
          f"({model['train_games']} games), calibrated on {model['calibration_seasons'][-1]}.")
    if "agent" in info:
        agent = info["agent"]
        print(f"Agent trained on priced seasons {agent['train_seasons'][0]} to {agent['train_seasons'][-1]}, "
              f"checked on {agent['validation_season']}, {agent['rounds_run']} learning rounds.")
    return 0


def cmd_predict(args) -> int:
    from . import predict
    date = args.date or dt.date.today()
    table = predict.predict_date(date, write_log=not args.no_log)
    print(f"\nMLB games on {date}  (paper trading only, no real bets are placed)\n")
    header = (f"{'Away':<5}{'Home':<6}{'Starters':<38}{'Lineups':<10}{'Home win':>9}{'Market':>8}{'Best edge':>13}  "
              f"{'Agent':<12}{'Agent lean':<13}{'Rule':<12}")
    print(header)
    print("-" * len(header))
    for row in table.itertuples(index=False):
        starters = f"{row.away_sp[:17]} v {row.home_sp[:17]}"
        market = "none" if pd.isna(row.market_home_prob) else _pct(row.market_home_prob)
        if pd.isna(row.home_dec):
            best = "n/a"
        else:
            side, value = ("home", row.edge_home) if row.edge_home >= row.edge_away else ("away", row.edge_away)
            best = f"{side} {100 * value:+.1f}%"

        def show(side, stake):
            return f"{side} {100 * stake:.1f}%" if side in ("home", "away") else side
        lineups = "posted" if row.lineups_posted else "guessed"
        print(f"{row.away:<5}{row.home:<6}{starters:<38}{lineups:<10}{_pct(row.p_home):>9}{market:>8}{best:>13}  "
              f"{show(row.agent_side, row.agent_stake):<12}{show(row.agent_lean_side, row.agent_lean_stake):<13}"
              f"{show(row.rule_side, row.rule_stake):<12}")
    print("\nAgent: its decision, which is to pass unless betting has been proven better on past seasons.")
    print("Agent lean: what its learned policy would bet with that safety bar off. For information only.")
    if not table["lineups_posted"].all():
        print("Lineups marked \"guessed\" are taken from the team's recent games. Run `mlb update` and predict "
              "again closer to game time, once lineups are posted.")
    if not table["made_before_game"].all():
        print("\nNote: some of these games are already finished. Those rows are replays, not forecasts.")
    if not args.no_log:
        print(f"\nLogged {len(table)} predictions. Run `mlb grade` after the games.")
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"Saved to {args.csv}")
    return 0


def cmd_grade(args) -> int:
    from . import predict
    result = predict.grade()
    print(f"Graded {result['graded_now']} predictions. Still waiting on {result['waiting']}.")
    return 0


def cmd_learn(args) -> int:
    from . import predict
    entry = predict.learn(rounds=args.rounds)
    print(f"The agent learned from {entry['games_learned_from']} games (newest {entry['newest_game']}).")
    print(f"On the {entry['games_held_back']} most recent games it was not shown, its average result went from "
          f"{entry['held_back_reward_before']:+.4f} to {entry['held_back_reward_after']:+.4f} (percent of bankroll per game). "
          f"It bets on {_pct(entry['bet_rate_after'])} of games, was {_pct(entry['bet_rate_before'])}.")
    if entry["kept_never_bet_policy"]:
        print("No betting policy was proven better than not betting, so the agent will keep passing.")
    return 0


def cmd_scorecard(args) -> int:
    from . import predict
    card = predict.scorecard()
    if args.json:
        print(json.dumps(card, indent=2, default=float))
        return 0
    print(f"Logged games: {card['logged_games']}. Graded: {card['graded_games']}. Waiting: {card['waiting_for_results']}.")
    for label, title in (("made_before_game", "Predictions made before the game (the real record)"),
                         ("replayed_after_game", "Replays logged after the game (not a real record)")):
        block = card[label]
        print(f"\n{title}")
        if not block.get("games", 1):
            print("  none yet")
            continue
        model = block["model"]
        print(f"  Model:  {model['games']} games, {_pct(model['accuracy'])} correct, log loss {model['log_loss']:.4f}")
        if "market" in block:
            market = block["market"]
            print(f"  Market: {market['games']} games, {_pct(market['accuracy'])} correct, log loss {market['log_loss']:.4f}")
            for who, name in (("agent", "GRPO agent"), ("rule", "Edge rule")):
                run = block[f"{who}_paper_bets"]
                line = (f"  {name}: {run['bets']} paper bets, return per unit staked {_pct(run['roi'])}, "
                        f"bankroll {run['start_bankroll']:.0f} to {run['final_bankroll']:.0f}")
                if run["bets"] and run.get("value_against_close") == run.get("value_against_close"):
                    line += f", price taken vs fair closing price {_pct(run['value_against_close'], 2)}"
                print(line)
    return 0


def cmd_page(args) -> int:
    from . import report
    path = report.write_page(Path(args.out))
    print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"mlb-predict {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress messages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download all raw data")
    p.add_argument("--no-odds", action="store_true", help="skip the betting odds (much faster)")
    p.add_argument("--refresh", action="store_true", help="download again even if a file is already there")
    p.set_defaults(func=cmd_fetch)

    sub.add_parser("update", help="daily refresh of results, starters and prices").set_defaults(func=cmd_update)
    sub.add_parser("build", help="make and check the clean tables").set_defaults(func=cmd_build)
    p = sub.add_parser("backtest", help="season-by-season test of model, market and betting")
    p.add_argument("--groups", action="store_true",
                   help="also measure what each group of player-level inputs is worth (slow)")
    p.set_defaults(func=cmd_backtest)
    sub.add_parser("train", help="fit the model and agent used for new predictions").set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="win chances and paper bets for one date")
    p.add_argument("--date", type=_date, help="YYYY-MM-DD (default: today)")
    p.add_argument("--no-log", action="store_true", help="do not write these predictions to the log")
    p.add_argument("--csv", help="also save the table to this file")
    p.set_defaults(func=cmd_predict)

    sub.add_parser("grade", help="score logged predictions whose games are final").set_defaults(func=cmd_grade)

    p = sub.add_parser("learn", help="let the agent learn from the newest results")
    p.add_argument("--rounds", type=int, default=20, help="number of GRPO learning rounds (default 20)")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("scorecard", help="track record of logged predictions")
    p.add_argument("--json", action="store_true", help="print the full numbers as JSON")
    p.set_defaults(func=cmd_scorecard)

    p = sub.add_parser("page", help="write the HTML results page")
    p.add_argument("--out", default=str(REPORTS / "index.html"))
    p.set_defaults(func=cmd_page)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
