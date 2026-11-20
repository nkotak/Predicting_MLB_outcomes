"""Readable summaries of the backtest: console text, Markdown and the HTML page."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

STRATEGY_NAMES = {
    "grpo_agent": "GRPO agent",
    "edge_rule": "Edge rule (3%+, quarter Kelly)",
    "always_home": "Always the home team",
    "always_favourite": "Always the favourite",
    "always_underdog": "Always the underdog",
}
FORECAST_NAMES = {
    "model": "Model",
    "linear_only": "Logistic regression part only",
    "trees_only": "Boosted trees part only",
    "team_rating_only": "Team rating (Elo) only",
    "always_home": "Always pick the home team",
}


def pct(value, digits: int = 1) -> str:
    return "n/a" if value is None or value != value else f"{100 * value:.{digits}f}%"


def _span(block: dict, digits: int = 4, scale: float = 1.0, unit: str = "") -> str:
    return (f"{scale * block['diff']:+.{digits}f}{unit} "
            f"(95% range {scale * block['low']:+.{digits}f} to {scale * block['high']:+.{digits}f})")


def backtest_text(result: dict) -> str:
    """A short summary for the terminal."""
    overall = result["forecast"]["overall"]
    lines = [
        f"Tested on {result['games_tested']:,} games, seasons {result['test_seasons'][0]} to {result['test_seasons'][-1]}.",
        "Each season was predicted by a model that had never seen it.",
        "",
        f"{'':<32}{'Correct':>9}{'Log loss':>10}",
    ]
    for key, name in FORECAST_NAMES.items():
        lines.append(f"{name:<32}{pct(overall[key]['accuracy']):>9}{overall[key]['log_loss']:>10.4f}")
    if "market" in overall:
        lines += [
            "",
            f"On the {overall['market']['games']:,} games with betting odds:",
            f"{'Betting market (closing price)':<32}{pct(overall['market']['accuracy']):>9}{overall['market']['log_loss']:>10.4f}",
            f"{'Model, same games':<32}{pct(overall['model_on_priced_games']['accuracy']):>9}{overall['model_on_priced_games']['log_loss']:>10.4f}",
            "Model log loss minus market log loss: " + _span(overall["model_minus_market_log_loss"])
            + ". Above zero means the market is the better forecaster.",
        ]
    betting = result.get("betting", {})
    if betting.get("seasons"):
        lines += ["", f"Paper betting, seasons {betting['seasons'][0]} to {betting['seasons'][-1]} "
                      f"({betting['games']:,} games with odds):",
                  f"{'':<32}{'Bets':>7}{'Won':>8}{'Return/unit':>13}{'Bankroll':>11}"]
        for key, name in STRATEGY_NAMES.items():
            run = betting["overall"].get(key)
            if run:
                lines.append(f"{name:<32}{run['bets']:>7}{pct(run['win_rate']):>8}{pct(run['roi']):>13}"
                             f"{pct(run['growth']):>11}")
    return "\n".join(lines)


def backtest_markdown(result: dict) -> str:
    """The full backtest as a Markdown report."""
    forecast, betting = result["forecast"], result.get("betting", {})
    overall = forecast["overall"]
    out = [
        "# Backtest results",
        "",
        f"Generated {dt.date.today().isoformat()} by `mlb backtest`. Do not edit by hand.",
        "",
        f"**{result['games_tested']:,} games**, seasons {result['test_seasons'][0]} to {result['test_seasons'][-1]}. "
        "Every season was predicted by a model that was trained and calibrated only on earlier seasons.",
        "",
        "## How good are the forecasts?",
        "",
        "\"Correct\" is the share of games where the side given the higher chance won. "
        "\"Log loss\" scores the probabilities themselves and lower is better. "
        "A coin flip scores 0.6931.",
        "",
        "| Forecaster | Games | Correct | Log loss | Brier | Calibration error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, name in FORECAST_NAMES.items():
        s = overall[key]
        out.append(f"| {name} | {s['games']:,} | {pct(s['accuracy'])} | {s['log_loss']:.4f} | {s['brier']:.4f} | "
                   f"{pct(s['calibration_error'], 2)} |")
    out += [
        "",
        "Model compared with always picking the home team: accuracy "
        + _span(overall["model_minus_always_home_accuracy"], 1, 100.0, " points")
        + ", log loss " + _span(overall["model_minus_always_home_log_loss"]) + ".",
    ]
    if "market" in overall:
        m, mp = overall["market"], overall["model_on_priced_games"]
        out += [
            "",
            "## Against the betting market",
            "",
            f"Closing prices are available for {m['games']:,} of the tested games "
            f"({pct(overall['odds_coverage'])}). On those games:",
            "",
            "| Forecaster | Correct | Log loss | Calibration error |",
            "|---|---:|---:|---:|",
            f"| Betting market (closing price, bookmaker's cut removed) | {pct(m['accuracy'])} | {m['log_loss']:.4f} | {pct(m['calibration_error'], 2)} |",
            f"| Model | {pct(mp['accuracy'])} | {mp['log_loss']:.4f} | {pct(mp['calibration_error'], 2)} |",
            "",
            "Model log loss minus market log loss: " + _span(overall["model_minus_market_log_loss"])
            + ". A value above zero means the market is the better forecaster.",
        ]
    out += ["", "## Season by season", "",
            "| Season | Games | Always home | Model | Market | Model log loss | Market log loss |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for season, block in forecast["by_season"].items():
        market = block.get("market")
        out.append(
            f"| {season} | {block['model']['games']:,} | {pct(block['always_home']['accuracy'])} | "
            f"{pct(block['model']['accuracy'])} | {pct(market['accuracy']) if market else 'no odds'} | "
            f"{block['model']['log_loss']:.4f} | {f'{market['log_loss']:.4f}' if market else ''} |")
    if betting.get("seasons"):
        out += [
            "", "## Paper betting", "",
            f"Seasons {betting['seasons'][0]} to {betting['seasons'][-1]}, {betting['games']:,} games with odds. "
            "No real money was involved. Bets are placed at the closing price. "
            "\"Return per unit\" is the profit per 1 unit staked if every bet were the same size, "
            "with a 95% range from resampling the bets. "
            "\"Bankroll\" starts at 1,000 units and uses each strategy's own stake sizes.",
            "",
            "| Strategy | Bets | Won | Avg price | Return per unit | 95% range | Final bankroll | Worst drop |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for key, name in STRATEGY_NAMES.items():
            run = betting["overall"].get(key)
            if not run:
                continue
            if run["bets"] == 0:
                out.append(f"| {name} | 0 | | | | | {run['final_bankroll']:,.0f} | |")
                continue
            out.append(f"| {name} | {run['bets']:,} | {pct(run['win_rate'])} | {run['average_price']:.2f} | "
                       f"{pct(run['roi'])} | {pct(run['roi_low'])} to {pct(run['roi_high'])} | "
                       f"{run['final_bankroll']:,.0f} | {pct(run['max_drawdown'])} |")
        out += ["", "### The agent, season by season", "",
                "| Season | Trained on | Bets | Share of games bet | Return per unit | Edge rule return per unit |",
                "|---|---|---:|---:|---:|---:|"]
        for season, block in betting["by_season"].items():
            agent, rule = block.get("grpo_agent"), block.get("edge_rule")
            note = betting["agent_training"].get(season) or betting["agent_training"].get(str(season)) or {}
            trained = note.get("train_seasons") or []
            span = f"{trained[0]} to {trained[-1]}" if trained else ""
            out.append(f"| {season} | {span} | {agent['bets'] if agent else 0:,} | {pct(note.get('bet_rate'))} | "
                       f"{pct(agent['roi']) if agent and agent['bets'] else 'no bets'} | "
                       f"{pct(rule['roi']) if rule and rule['bets'] else 'no bets'} |")
    out += ["", "## Inputs the model is allowed to see", "",
            ", ".join(f"`{name}`" for name in result["inputs"]), "",
            "---",
            "The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
            "Interested parties may contact Retrosheet at www.retrosheet.org."]
    return "\n".join(out) + "\n"


def load_backtest(path: Path = Path("reports/backtest.json")) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `mlb backtest` first.")
    return json.loads(path.read_text())
