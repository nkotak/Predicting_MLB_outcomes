"""Readable summaries of the backtest: console text, Markdown and the HTML page."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

STRATEGY_NAMES = {
    "grpo_agent": "GRPO agent",
    "grpo_agent_safety_off": "GRPO agent, safety bar off",
    "edge_rule": "Edge rule (3%+, quarter Kelly)",
    "always_home": "Always the home team",
    "always_favourite": "Always the favourite",
    "always_underdog": "Always the underdog",
}
GROUP_NAMES = {
    "lineup": "The nine hitters actually starting",
    "left_right": "Left/right matchups, hitters against the starter",
    "history": "Each hitter's past record against this starter",
    "starter_stuff": "Starter's pitches: swings and misses, speed, ground balls",
    "bullpen": "Bullpen quality and tiredness",
    "defense": "Team fielding",
    "statcast": "Statcast: contact quality, expected hitting, fielding range, speed",
    "injuries": "Players on the injured list",
    "recent_form": "Hot and cold streaks: the last three weeks against the longer record",
    "age": "Age of the hitters and the starter",
    "setting": "Ballpark, weather and umpire",
    "expected_runs": "Expected runs, everything rolled together",
}
FORECAST_NAMES = {
    "model": "Model",
    "linear_only": "Logistic regression part only",
    "trees_only": "Boosted trees part only",
    "margin_only": "Run margin part only",
    "team_rating_only": "Team rating (Elo) only",
    "always_home": "Always pick the home team",
}


def pct(value, digits: int = 1) -> str:
    if value is None or value != value:
        return "n/a"
    text = f"{100 * value:.{digits}f}%"
    return text[1:] if text.startswith("-") and float(text[:-1]) == 0 else text  # never "-0.0%"


def season_span(seasons: list) -> str:
    """'season 2026' or 'seasons 2024 to 2026'."""
    return f"season {seasons[0]}" if seasons[0] == seasons[-1] else f"seasons {seasons[0]} to {seasons[-1]}"


def _span(block: dict, digits: int = 4, scale: float = 1.0, unit: str = "") -> str:
    return (f"{scale * block['diff']:+.{digits}f}{unit} "
            f"(95% range {scale * block['low']:+.{digits}f} to {scale * block['high']:+.{digits}f})")


def _forecast_rows(overall: dict) -> list[tuple[str, dict]]:
    """Every forecaster to list, in order, with the name to show for it."""
    rows = [(name, overall[key]) for key, name in FORECAST_NAMES.items() if key in overall]
    if "price_taught_only" in overall:
        since = overall["price_taught_seasons"][0]
        rows.append((f"Price-taught part only ({since} on)", overall["price_taught_only"]))
        rows.append((f"Whole model, same seasons ({since} on)", overall["model_on_price_taught_seasons"]))
    return rows


def _verdict(block: dict) -> str:
    """Plain reading of a "helps by" number and its range."""
    if block["low"] > 0:
        return "helps"
    if block["high"] < 0:
        return "hurts"
    return "too small to tell from chance"


def _thousandths(block: dict) -> str:
    return f"{1000 * block['diff']:+.2f} ({1000 * block['low']:+.2f} to {1000 * block['high']:+.2f})"


def player_detail_sentence(result: dict) -> str:
    """One sentence on what the player-level inputs did to the score."""
    gap = result.get("player_detail", {}).get("model_minus_team_only_log_loss")
    if not gap:
        return ""
    if gap["high"] < 0:
        verdict = "The player detail made the forecasts better, by more than chance"
    elif gap["low"] > 0:
        verdict = "The player detail made the forecasts worse"
    else:
        verdict = "The player detail changed the forecasts by an amount that cannot be told apart from chance"
    return f"{verdict}: log loss changed by {_span(gap)}. Below zero is better."


def market_mix_sentence(block: dict, price: str) -> str:
    """One sentence on whether the model adds anything to a market price."""
    if not block.get("games"):
        return ""
    gap = block["mix_minus_market"]
    if gap["high"] < 0:
        verdict = "the model adds real information to this price"
    elif gap["low"] > 0:
        verdict = "adding the model made the price worse"
    else:
        verdict = "the model adds nothing to this price that can be told apart from chance"
    share = block.get("model_share")
    weight = "" if share is None or share != share else f" The mix gives the model {pct(max(share, 0.0), 0)} of the weight."
    return (f"{price.capitalize()} price, {block['games']:,} games: market alone {block['log_loss_market']:.4f}, "
            f"market mixed with the model {block['log_loss_mix']:.4f}, change {_span(gap)}, so {verdict}.{weight}")


def best_price_sentence(result: dict) -> str:
    """What taking the best price across sportsbooks did, on games several books quoted."""
    middle, best = result.get("best_price_middle", {}), result.get("best_price", {})
    if not (middle.get("seasons") and best.get("seasons")):
        return ""
    names = ("always_home", "always_favourite", "always_underdog")
    cut = lambda report: -sum(report["overall"][k]["roi"] for k in names) / len(names)
    rule_m, rule_b = middle["overall"].get("edge_rule"), best["overall"].get("edge_rule")
    text = (f"On the {best['games']:,} games that three or more sportsbooks quoted ({season_span(best['seasons'])}), "
            f"betting blind costs about {pct(cut(middle))} per unit staked at the middle price and about "
            f"{pct(cut(best))} at the best price on offer.")
    if rule_m and rule_b and rule_m["bets"] and rule_b["bets"]:
        text += (f" The model's own picks (the 3% edge rule) returned {pct(rule_m['roi'])} at the middle price and "
                 f"{pct(rule_b['roi'])} at the best price (95% range {pct(rule_b['roi_low'])} to {pct(rule_b['roi_high'])}).")
    return text


def early_sentence(result: dict) -> str:
    """Plain reading of the early-price paper bets on the model's picks."""
    rule = result.get("early_prices", {}).get("overall", {}).get("edge_rule")
    if not (rule and rule.get("bets")):
        return ""
    text = (f"Paper bets on the model's picks at opening prices returned {pct(rule['roi'])} per unit staked "
            f"({rule['bets']:,} bets, 95% range {pct(rule['roi_low'])} to {pct(rule['roi_high'])}).")
    if rule["roi_low"] > 0:
        return text + (" That range is above zero, the first sign of a real edge. It covers few seasons, so treat "
                       "it as a lead to follow and not as proof.")
    if rule["roi_high"] < 0:
        return text + " That is a loss, by more than chance."
    return text + " That range includes zero, so it could be luck either way."


def _model_story_markdown(result: dict) -> list[str]:
    """Sections on player detail, input groups, the market mix and price shopping."""
    overall = result["forecast"]["overall"]
    out: list[str] = []
    if "team_only" in overall:
        team, full = overall["team_only"], overall["model"]
        out += [
            "", "## What the player-by-player detail added", "",
            f"The team view describes each club as a whole and each starter from his box scores "
            f"({result.get('team_inputs', 30)} inputs). The player view adds the nine hitters actually starting, "
            f"left/right matchups, the bullpen and more ({result.get('player_inputs', 0)} inputs).",
            "",
            "| Version | Correct | Log loss | Calibration error |",
            "|---|---:|---:|---:|",
            f"| Team view only | {pct(team['accuracy'])} | {team['log_loss']:.4f} | {pct(team['calibration_error'], 2)} |",
            f"| Team view plus player detail (the model) | {pct(full['accuracy'])} | {full['log_loss']:.4f} | "
            f"{pct(full['calibration_error'], 2)} |",
        ]
        guessed = overall.get("model_with_guessed_lineups")
        if guessed:
            out.append(f"| The model with lineups guessed that morning | {pct(guessed['accuracy'])} | "
                       f"{guessed['log_loss']:.4f} | {pct(guessed['calibration_error'], 2)} |")
        out += ["", player_detail_sentence(result)]
    groups = result.get("input_groups")
    if groups:
        out += [
            "", "### What each group of inputs is worth", "",
            "Each group was taken out of the model (or put in, if it had been left out) and the whole "
            "season-by-season test was run again. \"Helps by\" is how much better the log loss is with the group "
            "than without it, in thousandths, with a 95% range. A range that includes zero means the effect "
            "cannot be told apart from chance.",
            "",
            "| Group of inputs | In the model | Helps by (95% range) | Verdict |",
            "|---|---|---:|---|",
        ]
        for key, block in sorted(groups.items(), key=lambda item: -item[1]["helps_by"]["diff"]):
            out.append(f"| {GROUP_NAMES.get(key, key)} | {'yes' if block['in_model'] else 'no, tested and left out'} | "
                       f"{_thousandths(block['helps_by'])} | {_verdict(block['helps_by'])} |")
    mix = result.get("market_blend", {})
    lines = [market_mix_sentence(mix.get("closing", {}), "closing"), market_mix_sentence(mix.get("opening", {}), "opening")]
    if any(lines):
        out += [
            "", "## Does the model know anything the market does not?", "",
            "A simple mix of the market's chance and the model's chance was fitted on earlier seasons and scored "
            "on the next one. If the mix scores no better than the market on its own, the model holds nothing the "
            "price does not already contain. For opening prices the model uses lineups guessed that morning.",
            "",
        ] + [f"- {line}" for line in lines if line]
    best = result.get("best_price", {})
    if best.get("seasons"):
        out += [
            "", "## Shopping for the best price", "",
            "The betting tests use the middle price across sportsbooks. Someone with several accounts can take the "
            f"best price on offer instead. This test uses the {best['games']:,} games "
            f"({season_span(best['seasons'])}) that at least three sportsbooks quoted. From 2024 the free feed "
            "carries one sportsbook only, so there is nothing to shop between in those seasons.",
            "", "At the best closing price:", "",
        ] + _strategy_table(best, with_close=False)
        middle = result.get("best_price_middle", {})
        if middle.get("seasons"):
            out += ["", "The same games at the middle closing price:", ""] + _strategy_table(middle, with_close=False)
        out += ["", best_price_sentence(result)]
    return out


def backtest_text(result: dict) -> str:
    """A short summary for the terminal."""
    overall = result["forecast"]["overall"]
    lines = [
        f"Tested on {result['games_tested']:,} games, seasons {result['test_seasons'][0]} to {result['test_seasons'][-1]}.",
        "Each season was predicted by a model that had never seen it.",
        "",
        f"{'':<40}{'Correct':>9}{'Log loss':>10}",
    ]
    for name, block in _forecast_rows(overall):
        lines.append(f"{name:<40}{pct(block['accuracy']):>9}{block['log_loss']:>10.4f}")
    if "market" in overall:
        lines += [
            "",
            f"On the {overall['market']['games']:,} games with betting odds:",
            f"{'Betting market (closing price)':<40}{pct(overall['market']['accuracy']):>9}{overall['market']['log_loss']:>10.4f}",
            f"{'Model, same games':<40}{pct(overall['model_on_priced_games']['accuracy']):>9}{overall['model_on_priced_games']['log_loss']:>10.4f}",
            "Model log loss minus market log loss: " + _span(overall["model_minus_market_log_loss"])
            + ". Above zero means the market is the better forecaster.",
        ]
    if "team_only" in overall:
        lines += ["", f"{'Model without player detail':<40}{pct(overall['team_only']['accuracy']):>9}"
                      f"{overall['team_only']['log_loss']:>10.4f}", player_detail_sentence(result)]
    mix = result.get("market_blend", {})
    for line in (market_mix_sentence(mix.get("closing", {}), "closing"), market_mix_sentence(mix.get("opening", {}), "opening")):
        if line:
            lines += ["", line]
    for key, title in (("betting", "Paper betting at closing prices"),
                       ("best_price_middle", "The middle closing price, on games quoted by three or more sportsbooks"),
                       ("best_price", "The best closing price across sportsbooks, on those same games"),
                       ("early_prices", "Paper betting at opening prices")):
        betting = result.get(key, {})
        if not betting.get("seasons"):
            continue
        lines += ["", f"{title}, {season_span(betting['seasons'])} "
                      f"({betting['games']:,} games):",
                  f"{'':<40}{'Bets':>7}{'Won':>8}{'Return/unit':>13}{'Vs close':>10}"]
        for name_key, name in STRATEGY_NAMES.items():
            run = betting["overall"].get(name_key)
            if run:
                lines.append(f"{name:<40}{run['bets']:>7}{pct(run['win_rate']):>8}{pct(run['roi']):>13}"
                             f"{pct(run['value_against_close']):>10}")
    move = result.get("early_prices", {}).get("line_movement", {})
    if move.get("games"):
        lines += ["", f"Opening prices ({move['games']:,} games): the price moved toward the model's view. "
                      f"Link between the two: {move['correlation']:.2f} "
                      f"(0 would mean no link, plus or minus {move['correlation_margin']:.2f} is noise)."]
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
    for name, s in _forecast_rows(overall):
        out.append(f"| {name} | {s['games']:,} | {pct(s['accuracy'])} | {s['log_loss']:.4f} | {s['brier']:.4f} | "
                   f"{pct(s['calibration_error'], 2)} |")
    if "price_taught_only" in overall:
        out += ["",
                "The price-taught part learns from what closing prices said about past games. It needs a few "
                f"seasons of prices first, so it is in use from {overall['price_taught_seasons'][0]} on and is "
                "scored on those seasons, next to the whole model on the same seasons. It is never shown the "
                "price of the game it is predicting."]
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
    out += _model_story_markdown(result)
    out += ["", "## Season by season", "",
            "| Season | Games | Always home | Model | Market | Model log loss | Market log loss |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for season, block in forecast["by_season"].items():
        market = block.get("market")
        market_loss = f"{market['log_loss']:.4f}" if market else ""
        out.append(
            f"| {season} | {block['model']['games']:,} | {pct(block['always_home']['accuracy'])} | "
            f"{pct(block['model']['accuracy'])} | {pct(market['accuracy']) if market else 'no odds'} | "
            f"{block['model']['log_loss']:.4f} | {market_loss} |")
    if betting.get("seasons"):
        out += [
            "", "## Paper betting at closing prices", "",
            f"{season_span(betting['seasons']).capitalize()}, {betting['games']:,} games with odds. "
            "No real money was involved. This is the hardest test, because the closing price already "
            "contains everything the market learned that day. "
            "\"Return per unit\" is the profit per 1 unit staked if every bet were the same size, "
            "with a 95% range from resampling the bets. "
            "\"Bankroll\" starts at 1,000 units and uses each strategy's own stake sizes.",
            "",
        ]
        out += _strategy_table(betting, with_close=False)
        out += ["", _safety_off_sentence(betting)]
        out += ["", "### The agent, season by season", "",
                "\"Kept never-bet\" means that after training, no version of the agent was proven better than "
                "not betting on the held-back season, so it passed on every game.",
                "",
                "| Season | Trained on | Checked on | Result of training | Bets | Return per unit | Edge rule return per unit |",
                "|---|---|---|---|---:|---:|---:|"]
        out += _agent_rows(betting)
        out += ["", _learning_sentence(betting)]
    early = result.get("early_prices", {})
    move = early.get("line_movement", {})
    if move.get("games"):
        out += [
            "", "## Betting early, at opening prices", "",
            f"Opening prices are available for {move['games']:,} tested games "
            f"({season_span(move['seasons'])}).",
            "",
            f"- **The price moves toward the model.** For each point the model sat away from the opening price, "
            f"the price moved {move['slope']:.2f} of a point in the model's direction by game time. "
            f"The link between the two is {move['correlation']:.2f} on a scale where 0 means none "
            f"(anything inside plus or minus {move['correlation_margin']:.2f} would be noise).",
            f"- **Opening prices are softer than closing prices.** Log loss: opening price "
            f"{move['log_loss_opening_price']:.4f}, closing price {move['log_loss_closing_price']:.4f}, "
            f"model {move['log_loss_model']:.4f}. The price moved {move['average_move_points']:.1f} points "
            "on average between open and close.",
            "",
            "For this section the model was not shown the real lineups. It used lineups guessed from each "
            "team's recent games, as a bettor would have to in the morning.",
            "",
            "One caution. The model is still given the pitcher who actually started. When an opening price was set "
            "before a late change of starter, the model knew something that nobody could have known at the time. "
            "That flatters these numbers by an unknown amount. The prediction log, which records the price at "
            "the moment a prediction is made, is the fair way to measure this from now on.",
        ]
        if early.get("seasons"):
            out += [
                "",
                f"Paper bets at the opening price, {early['games']:,} games "
                f"({season_span(early['seasons'])}). "
                "\"Value against close\" compares the price taken with the fair closing price. "
                "Above zero means the bets beat the closing line, which is the usual early sign of a real edge.",
                "",
            ]
            out += _strategy_table(early, with_close=True)
            out += ["", "| Season | Trained on | Checked on | Result of training | Bets | Return per unit | Edge rule return per unit |",
                    "|---|---|---|---|---:|---:|---:|"]
            out += _agent_rows(early)
            out += ["", early_sentence(result)]
    out += ["", "## Inputs the model is allowed to see", "",
            ", ".join(f"`{name}`" for name in result["inputs"]), "",
            "---",
            "The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
            "Interested parties may contact Retrosheet at www.retrosheet.org."]
    return "\n".join(out) + "\n"


def _strategy_table(betting: dict, *, with_close: bool) -> list[str]:
    head = "| Strategy | Bets | Won | Avg price | Return per unit | 95% range |"
    rule = "|---|---:|---:|---:|---:|---|"
    head += " Value against close |" if with_close else " Final bankroll | Worst drop |"
    rule += "---:|" if with_close else "---:|---:|"
    rows = [head, rule]
    for key, name in STRATEGY_NAMES.items():
        run = betting["overall"].get(key)
        if not run:
            continue
        if run["bets"] == 0:
            tail = " |" if with_close else f" {run['final_bankroll']:,.0f} | |"
            rows.append(f"| {name} | 0 | | | no bets | |{tail}")
            continue
        tail = (f" {pct(run['value_against_close'], 2)} |" if with_close
                else f" {run['final_bankroll']:,.0f} | {pct(run['max_drawdown'])} |")
        rows.append(f"| {name} | {run['bets']:,} | {pct(run['win_rate'])} | {run['average_price']:.2f} | "
                    f"{pct(run['roi'])} | {pct(run['roi_low'])} to {pct(run['roi_high'])} |{tail}")
    return rows


def _safety_off_sentence(betting: dict) -> str:
    """What the agent's learned betting would have done without its safety bar."""
    raw = betting.get("overall", {}).get("grpo_agent_safety_off")
    if not raw:
        return ""
    if raw["bets"] == 0:
        return ("\"Safety bar off\" shows the bets the agent's learned policy would have placed if it were not "
                "required to prove itself first. Even then it placed no bets: by the end of training it had "
                "taught itself to pass on everything.")
    return ("\"Safety bar off\" shows the bets the agent's learned policy would have placed if it were not "
            f"required to prove itself first: {raw['bets']:,} paper bets, {pct(raw['win_rate'])} won, "
            f"{pct(raw['roi'])} return per unit staked (95% range {pct(raw['roi_low'])} to {pct(raw['roi_high'])}). "
            "The safety bar is what stopped those bets from being recommended.")


def _learning_sentence(betting: dict) -> str:
    """One plain sentence on how the agent's behaviour changed while it trained."""
    notes = [n for n in betting.get("agent_training", {}).values() if "bet_rate_first_round" in n]
    if not notes:
        return ""
    first = sum(n["bet_rate_first_round"] for n in notes) / len(notes)
    last = sum(n["bet_rate_last_round"] for n in notes) / len(notes)
    held = sum(n["validation_reward_last_round"] for n in notes) / len(notes)
    verdict = ("did not make money" if held <= 0 else "made a little, though not enough to rule out luck")
    return (f"What the learning looked like: averaged over these seasons, the agent started out betting on "
            f"{pct(first, 0)} of its training games and ended up betting on {pct(last, 0)}. "
            f"On the held-back season, what it still wanted to bet on {verdict}.")


def _agent_rows(betting: dict) -> list[str]:
    rows = []
    for season, block in betting["by_season"].items():
        agent, rule = block.get("grpo_agent"), block.get("edge_rule")
        note = betting["agent_training"].get(season) or betting["agent_training"].get(str(season)) or {}
        trained = note.get("train_seasons") or []
        span = "" if not trained else (str(trained[0]) if trained[0] == trained[-1] else f"{trained[0]} to {trained[-1]}")
        outcome = "kept never-bet" if note.get("kept_never_bet_policy") else "a betting policy was proven"
        rows.append(f"| {season} | {span} | {note.get('validation_season', '')} | {outcome} | "
                    f"{agent['bets'] if agent else 0:,} | "
                    f"{pct(agent['roi']) if agent and agent['bets'] else 'no bets'} | "
                    f"{pct(rule['roi']) if rule and rule['bets'] else 'no bets'} |")
    return rows


def load_backtest(path: Path = Path("reports/backtest.json")) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `mlb backtest` first.")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# The HTML results page
# ---------------------------------------------------------------------------

PAGE_CSS = """
:root{--bg:#f4f6f8;--surface:#ffffff;--ink:#101a26;--ink2:#465566;--muted:#6d7b8a;--line:#d9e0e7;
--accent:#0f5fb8;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0b7a43;--bad:#b3261e;--hero:#0a1422}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b1118;--surface:#131c26;--ink:#e8eef4;
--ink2:#b3c0cd;--muted:#8494a5;--line:#243140;--accent:#6fb1ff;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4cc38a;
--bad:#ff8a80;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0b1118;--surface:#131c26;--ink:#e8eef4;--ink2:#b3c0cd;--muted:#8494a5;--line:#243140;
--accent:#6fb1ff;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4cc38a;--bad:#ff8a80;color-scheme:dark}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding-left:max(20px,env(safe-area-inset-left,0px));
padding-right:max(20px,env(safe-area-inset-right,0px))}
.hero{background:var(--hero) center 78%/cover no-repeat;color:#fff;position:relative}
.hero::before{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(6,12,22,.62) 0%,rgba(6,12,22,.38) 55%,rgba(6,12,22,.82) 100%)}
.hero .wrap{position:relative;padding-top:64px;padding-bottom:40px;min-height:540px;display:flex;flex-direction:column}
.hero .stats{margin-top:auto}
.kicker{font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:#9fc6ff;margin:0 0 8px;font-weight:600}
h1{font-size:clamp(2.1rem,6vw,3.3rem);line-height:1.05;margin:0 0 14px;letter-spacing:-.01em}
.lede{font-size:1.12rem;max-width:58ch;color:#d6e2ef;margin:0 0 28px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.stat{background:rgba(8,16,28,.62);border:1px solid rgba(255,255,255,.2);border-radius:10px;padding:14px 16px;
-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px)}
.stat b{display:block;font-size:1.9rem;line-height:1.1;font-weight:650}
.stat span{font-size:.86rem;color:#c5d3e2}
section{padding:44px 0 8px}
h2{font-size:1.55rem;line-height:1.2;margin:0 0 10px;letter-spacing:-.005em}
p{margin:0 0 14px;max-width:70ch}.note{color:var(--ink2);font-size:.93rem}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;margin:18px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.93rem}
th{text-align:left;font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
padding:8px 12px 8px 0;border-bottom:1px solid var(--line);font-weight:600;white-space:nowrap}
td{padding:9px 12px 9px 0;border-bottom:1px solid var(--line);vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}.up{color:var(--good)}.down{color:var(--bad)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.88rem;color:var(--ink2);margin-bottom:6px}
.legend i{display:inline-block;width:20px;border-top:2px solid;vertical-align:middle;margin-right:6px}
svg text{font:12px system-ui,sans-serif;fill:var(--muted)}svg .grid{stroke:var(--line)}
.pill{display:inline-block;font-size:.76rem;font-weight:600;padding:2px 9px;border-radius:999px;
border:1px solid var(--line);color:var(--ink2);white-space:nowrap}
.pill.bet{border-color:var(--accent);color:var(--accent)}
.verdict{border-left:4px solid var(--accent);padding:4px 0 4px 16px;margin:18px 0;font-size:1.05rem}
footer{padding:36px 0 60px;color:var(--muted);font-size:.86rem}
details summary{cursor:pointer;color:var(--accent);font-size:.9rem;margin-top:8px}
"""


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def season_chart(by_season: dict) -> str:
    """Accuracy by season as an SVG line chart: model, always home, and the market where known."""
    seasons = [int(s) for s in by_season]

    def get(season: int) -> dict:  # keys are text after a round trip through JSON
        return by_season[str(season)] if str(season) in by_season else by_season[season]

    series = {
        "Model": ("var(--s1)", [get(s)["model"]["accuracy"] for s in seasons]),
        "Always the home team": ("var(--s2)", [get(s)["always_home"]["accuracy"] for s in seasons]),
        "Betting market": ("var(--s3)", [get(s)["market"]["accuracy"] if "market" in get(s)
                                         and get(s)["market"]["games"] >= 300 else None for s in seasons]),
    }
    width, height, left, right, top, bottom = 900, 320, 46, 16, 14, 30
    values = [v for _, vals in series.values() for v in vals if v is not None]
    low = min(0.50, (int(min(values) * 100) // 2 * 2) / 100)
    high = max(0.62, (int(max(values) * 100) // 2 * 2 + 2) / 100)
    def x(i): return left + i * (width - left - right) / max(len(seasons) - 1, 1)
    def y(v): return top + (high - v) / (high - low) * (height - top - bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Share of games called correctly, by season">']
    tick = low
    while tick <= high + 1e-9:
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        parts.append(f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick * 100:.0f}%</text>')
        tick += 0.02
    for i, season in enumerate(seasons):
        parts.append(f'<text x="{x(i):.1f}" y="{height - 8}" text-anchor="middle">{season}</text>')
    for colour, vals in series.values():
        run = []
        for i, value in enumerate(vals):
            if value is None:
                if len(run) > 1:
                    parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" points="{" ".join(run)}"/>')
                run = []
            else:
                run.append(f"{x(i):.1f},{y(value):.1f}")
        if len(run) > 1:
            parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" points="{" ".join(run)}"/>')
        for i, value in enumerate(vals):
            if value is not None:
                parts.append(f'<circle cx="{x(i):.1f}" cy="{y(value):.1f}" r="4.5" fill="{colour}" stroke="var(--surface)" stroke-width="2"/>')
    parts.append("</svg>")
    legend = "".join(f'<span><i style="border-color:{c}"></i>{_esc(n)}</span>' for n, (c, _) in series.items())
    return f'<div class="legend">{legend}</div><div class="scroll">{"".join(parts)}</div>'


def _signed(value, digits=1) -> str:
    if value is None or value != value:
        return "n/a"
    cls = "up" if value > 0 else "down" if value < 0 else ""
    return f'<span class="{cls}">{100 * value:+.{digits}f}%</span>'


def _strategy_rows_html(betting: dict, with_close: bool) -> str:
    rows = []
    for key, name in STRATEGY_NAMES.items():
        run = betting["overall"].get(key)
        if not run:
            continue
        if run["bets"] == 0:
            rows.append(f'<tr><td>{_esc(name)}</td><td class="n">0</td><td class="n"></td>'
                        f'<td class="n">no bets</td><td class="n"></td></tr>')
            continue
        last = _signed(run["value_against_close"], 2) if with_close else f'{run["final_bankroll"]:,.0f}'
        rows.append(f'<tr><td>{_esc(name)}</td><td class="n">{run["bets"]:,}</td><td class="n">{pct(run["win_rate"])}</td>'
                    f'<td class="n">{_signed(run["roi"])}</td><td class="n">{last}</td></tr>')
    return "".join(rows)


def render_page(result: dict, predictions=None, card: dict | None = None, hero_image: str | None = None,
                generated: str | None = None, *, fragment: bool = False) -> str:
    """The results page as one self-contained HTML document.

    With ``fragment`` the outer html, head and body tags are left out, for
    hosts that wrap the page in their own.
    """
    overall, betting = result["forecast"]["overall"], result.get("betting", {})
    early = result.get("early_prices", {})
    move = early.get("line_movement", {})
    model, home = overall["model"], overall["always_home"]
    market = overall.get("market")
    seasons = result["test_seasons"]
    hero_style = f' style="background-image:url({_esc(hero_image)})"' if hero_image else ""
    head = "<title>MLB Predict Results</title>" + f"<style>{PAGE_CSS}</style>"
    if fragment:
        head = '<meta name="claude:viewport-fit-sides" content="cover">' + head
    html = [
        head if fragment else
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">' + head + "</head><body>",
        f'<header class="hero"{hero_style}><div class="wrap">',
        '<p class="kicker">MLB Predict</p>',
        "<h1>Win chances for every game, tested against the betting market</h1>",
        f'<p class="lede">Every season from {seasons[0]} to {seasons[-1]} was predicted by a model that had never seen it. '
        "Paper trading only. Nothing here places a real bet.</p>",
        '<div class="stats">',
        f'<div class="stat"><b>{pct(model["accuracy"])}</b><span>of {model["games"]:,} games called correctly</span></div>',
        f'<div class="stat"><b>{pct(home["accuracy"])}</b><span>by always picking the home team</span></div>',
    ]
    if market:
        html.append(f'<div class="stat"><b>{pct(market["accuracy"])}</b><span>by the betting market, '
                    f'on the {market["games"]:,} games with odds</span></div>')
    agent_run = betting.get("overall", {}).get("grpo_agent") if betting.get("seasons") else None
    if agent_run is not None:
        html.append(f'<div class="stat"><b>{agent_run["bets"]:,}</b><span>paper bets the learning agent chose to place '
                    f'at closing prices</span></div>')
    html.append("</div></div></header><main class='wrap'>")

    html += ["<section><h2>How good are the forecasts?</h2>",
             "<p>The model gives each game a home win chance. \"Correct\" means the side it favoured won. "
             "Log loss scores the chances themselves, and lower is better. A coin flip scores 0.6931.</p>",
             '<div class="card">', season_chart(result["forecast"]["by_season"]),
             "<details><summary>Show the numbers</summary><div class='scroll'><table><thead><tr><th>Season</th>"
             "<th class='n'>Games</th><th class='n'>Always home</th><th class='n'>Model</th><th class='n'>Market</th>"
             "<th class='n'>Model log loss</th><th class='n'>Market log loss</th></tr></thead><tbody>"]
    for season, block in result["forecast"]["by_season"].items():
        mk = block.get("market")
        market_loss = f"{mk['log_loss']:.4f}" if mk else ""
        html.append(f'<tr><td>{season}</td><td class="n">{block["model"]["games"]:,}</td>'
                    f'<td class="n">{pct(block["always_home"]["accuracy"])}</td><td class="n">{pct(block["model"]["accuracy"])}</td>'
                    f'<td class="n">{pct(mk["accuracy"]) if mk else "no odds"}</td>'
                    f'<td class="n">{block["model"]["log_loss"]:.4f}</td><td class="n">{market_loss}</td></tr>')
    html.append("</tbody></table></div></details></div>")
    html.append('<div class="card scroll"><table><thead><tr><th>Forecaster</th><th class="n">Correct</th>'
                '<th class="n">Log loss</th><th class="n">Calibration error</th></tr></thead><tbody>')
    for name, s in _forecast_rows(overall):
        html.append(f'<tr><td>{_esc(name)}</td><td class="n">{pct(s["accuracy"])}</td><td class="n">{s["log_loss"]:.4f}</td>'
                    f'<td class="n">{pct(s["calibration_error"], 2)}</td></tr>')
    html.append("</tbody></table></div></section>")

    if market:
        gap = overall["model_minus_market_log_loss"]
        better = "the market is the better forecaster" if gap["low"] > 0 else (
            "the model is the better forecaster" if gap["high"] < 0 else "the two cannot be told apart")
        mp = overall["model_on_priced_games"]
        html += ["<section><h2>Against the betting market</h2>",
                 f"<p>On the {market['games']:,} games with closing odds, the market's price (with the bookmaker's cut removed) "
                 f"scores a log loss of <b>{market['log_loss']:.4f}</b> and the model scores <b>{mp['log_loss']:.4f}</b>.</p>",
                 f'<p class="verdict">The gap is {gap["diff"]:+.4f}, with a 95% range of {gap["low"]:+.4f} to {gap["high"]:+.4f}, '
                 f"so {better}.</p></section>"]

    if "team_only" in overall:
        team = overall["team_only"]
        guessed = overall.get("model_with_guessed_lineups")
        html += ["<section><h2>What the player-by-player detail added</h2>",
                 f"<p>The team view describes each club as a whole ({result.get('team_inputs', 30)} inputs). The player "
                 f"view adds the nine hitters actually starting, left/right matchups, the bullpen and more "
                 f"({result.get('player_inputs', 0)} inputs).</p>",
                 '<div class="card scroll"><table><thead><tr><th>Version</th><th class="n">Correct</th>'
                 '<th class="n">Log loss</th></tr></thead><tbody>',
                 f'<tr><td>Team view only</td><td class="n">{pct(team["accuracy"])}</td><td class="n">{team["log_loss"]:.4f}</td></tr>',
                 f'<tr><td>Team view plus player detail (the model)</td><td class="n">{pct(model["accuracy"])}</td>'
                 f'<td class="n">{model["log_loss"]:.4f}</td></tr>']
        if guessed:
            html.append(f'<tr><td>The model with lineups guessed that morning</td><td class="n">{pct(guessed["accuracy"])}</td>'
                        f'<td class="n">{guessed["log_loss"]:.4f}</td></tr>')
        html.append(f"</tbody></table></div><p class='verdict'>{_esc(player_detail_sentence(result))}</p>")
        groups = result.get("input_groups")
        if groups:
            html += ["<p>Each group of inputs was taken out of the model (or put in, if it had been left out) and the "
                     "whole test was run again. \"Helps by\" is the improvement in log loss with the group, in "
                     "thousandths. A range that includes zero cannot be told apart from chance.</p>",
                     '<div class="card scroll"><table><thead><tr><th>Group of inputs</th><th>In the model</th>'
                     '<th class="n">Helps by (95% range)</th><th>Verdict</th></tr></thead><tbody>']
            for key, block in sorted(groups.items(), key=lambda item: -item[1]["helps_by"]["diff"]):
                html.append(f'<tr><td>{_esc(GROUP_NAMES.get(key, key))}</td><td>{"yes" if block["in_model"] else "left out"}</td>'
                            f'<td class="n">{_esc(_thousandths(block["helps_by"]))}</td><td>{_esc(_verdict(block["helps_by"]))}</td></tr>')
            html.append("</tbody></table></div>")
        html.append("</section>")

    mix = result.get("market_blend", {})
    mix_lines = [market_mix_sentence(mix.get("closing", {}), "closing"), market_mix_sentence(mix.get("opening", {}), "opening")]
    if any(mix_lines):
        html += ["<section><h2>Does the model know anything the market does not?</h2>",
                 "<p>A simple mix of the market's chance and the model's chance was fitted on earlier seasons and "
                 "scored on the next one. If the mix scores no better than the market alone, the model holds nothing "
                 "the price does not already contain.</p>"]
        html += [f"<p class='note'>{_esc(line)}</p>" for line in mix_lines if line]
        html.append("</section>")

    if betting.get("seasons"):
        html += ["<section><h2>Paper betting at closing prices</h2>",
                 f"<p>{season_span(betting['seasons']).capitalize()}, {betting['games']:,} games. "
                 "The closing price is the hardest price to beat, because it already contains everything the market "
                 "learned that day. \"Return per unit\" is profit per 1 unit staked. Bankroll starts at 1,000.</p>",
                 '<div class="card scroll"><table><thead><tr><th>Strategy</th><th class="n">Bets</th><th class="n">Won</th>'
                 '<th class="n">Return per unit</th><th class="n">Final bankroll</th></tr></thead><tbody>',
                 _strategy_rows_html(betting, False), "</tbody></table></div>"]
        kept = [s for s, note in betting["agent_training"].items() if note.get("kept_never_bet_policy")]
        total = len(betting["agent_training"])
        html.append(f"<p class='note'>The learning agent (GRPO) is retrained before each season on earlier seasons only. "
                    f"In {len(kept)} of {total} seasons, no betting policy was proven better than not betting on the "
                    "held-back season, so it passed on every game. Passing is a real result: it means the evidence did "
                    f"not support betting at these prices.</p><p class='note'>{_esc(_learning_sentence(betting))}</p>"
                    f"<p class='note'>{_esc(_safety_off_sentence(betting))}</p></section>")

    best_price = result.get("best_price", {})
    if best_price.get("seasons"):
        html += ["<section><h2>Shopping for the best price</h2>",
                 "<p>The table above uses the middle price across sportsbooks. Someone with several accounts can take "
                 f"the best price on offer instead. This test uses the {best_price['games']:,} games "
                 f"({season_span(best_price['seasons'])}) that at least three sportsbooks quoted, at the best closing price:</p>",
                 '<div class="card scroll"><table><thead><tr><th>Strategy</th><th class="n">Bets</th><th class="n">Won</th>'
                 '<th class="n">Return per unit</th><th class="n">Final bankroll</th></tr></thead><tbody>',
                 _strategy_rows_html(best_price, False), "</tbody></table></div>",
                 f"<p class='verdict'>{_esc(best_price_sentence(result))}</p></section>"]

    if move.get("games"):
        html += ["<section><h2>Betting early, at opening prices</h2>",
                 f"<p>Opening prices exist for {move['games']:,} tested games ({season_span(move['seasons'])}). "
                 f"They are softer than closing prices, and <b>the price tends to move toward "
                 f"the model during the day</b>: for each point the model sat away from the opening price, the price "
                 f"moved {move['slope']:.2f} of a point in the model's direction by game time.</p>"]
        if early.get("seasons"):
            html += [f"<p>Paper bets at the opening price, {early['games']:,} games "
                     f"({season_span(early['seasons'])}). \"Against the close\" compares the price taken "
                     "with the fair closing price. Above zero means the bets beat the closing line.</p>",
                     '<div class="card scroll"><table><thead><tr><th>Strategy</th><th class="n">Bets</th><th class="n">Won</th>'
                     '<th class="n">Return per unit</th><th class="n">Against the close</th></tr></thead><tbody>',
                     _strategy_rows_html(early, True), "</tbody></table></div>"]
            html.append(f"<p class='verdict'>{_esc(early_sentence(result))}</p>")
        html.append("<p class='note'>For this section the model used lineups guessed from each team's recent games, as a "
                    "bettor would have to in the morning. One caution: it is still given the pitcher who actually "
                    "started. An opening price set before a late pitching change could not have known that, which "
                    "flatters these numbers by an unknown amount. The prediction log measures this fairly from now on.</p></section>")

    tables = predictions if isinstance(predictions, list) else ([] if predictions is None else [predictions])
    for predictions in [t for t in tables if len(t)]:
        day = str(predictions["date"].iloc[0])[:10]
        replay = not bool(predictions["made_before_game"].all())
        html += [f"<section><h2>Games on {day}</h2>",
                 "<p class='note'>" + ("These games are already finished, so this table is a replay and not a forecast."
                                       if replay else "Logged before the games were played.") + "</p>",
                 '<div class="card scroll"><table><thead><tr><th>Away</th><th>Home</th><th>Starters</th>'
                 '<th class="n">Home win</th><th class="n">Market</th><th class="n">Best edge</th><th>Agent</th></tr></thead><tbody>']
        for row in predictions.itertuples(index=False):
            has_price = row.home_dec == row.home_dec
            if has_price:
                side, value = ("home", row.edge_home) if row.edge_home >= row.edge_away else ("away", row.edge_away)
                best = f"{side} {_signed(value)}"
            else:
                best = "no price"
            agent = (f'<span class="pill bet">{row.agent_side} {100 * row.agent_stake:.1f}%</span>'
                     if row.agent_side in ("home", "away") else f'<span class="pill">{_esc(row.agent_side)}</span>')
            html.append(f'<tr><td>{_esc(row.away)}</td><td>{_esc(row.home)}</td>'
                        f'<td>{_esc(row.away_sp)} v {_esc(row.home_sp)}</td><td class="n">{pct(row.p_home)}</td>'
                        f'<td class="n">{pct(row.market_home_prob) if has_price else "none"}</td>'
                        f'<td class="n">{best}</td><td>{agent}</td></tr>')
        html.append("</tbody></table></div></section>")

    if card and card.get("graded_games"):
        live = card.get("made_before_game", {})
        if live.get("model"):
            html += ["<section><h2>Track record of logged predictions</h2>",
                     f"<p>{live['model']['games']:,} predictions were logged before their games and have been graded: "
                     f"{pct(live['model']['accuracy'])} correct, log loss {live['model']['log_loss']:.4f}.</p></section>"]

    html += ["<section><h2>How to read this page</h2>",
             "<p><b>No peeking.</b> Every input for a game is built only from earlier games, and the model can only see an "
             "approved list of inputs. A test fails if any input changes when that game's own result is changed.</p>",
             "<p><b>Calibration.</b> When the model says 60%, it should happen about 60% of the time. The calibration error "
             "column shows how far off that is on average.</p>",
             "<p><b>Why the market is hard to beat.</b> Prices already reflect line-ups, injuries, weather and the money of "
             "professional bettors, and the bookmaker takes a cut of about 4% on top. This model now sees line-ups, "
             "injuries and weather too, so what is left is the judgement of those bettors.</p></section>",
             "</main><footer><div class='wrap'>",
             f"<p>Generated {_esc(generated or dt.date.today().isoformat())} by <code>mlb page</code>. Paper trading only.</p>",
             "<p>The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
             "Interested parties may contact Retrosheet at www.retrosheet.org. Current-season results and schedules come "
             "from MLB's public feed, Statcast season tables from Baseball Savant, and betting prices from ESPN's "
             "public feed.</p>",
             "</div></footer>" + ("" if fragment else "</body></html>")]
    return "".join(html)


def write_page(out: Path = Path("reports/index.html"), *, date=None) -> Path:
    """Write the results page from the saved backtest and, when available, the prediction log."""
    result = load_backtest(out.parent / "backtest.json")
    predictions, card = None, None
    try:
        from . import predict
        log_frame = predict.latest_per_game(predict.read_log())
        if len(log_frame):
            card = predict.scorecard()
            from .betting import edge
            log_frame["edge_home"] = edge(log_frame["p_home"], log_frame["home_dec"])
            log_frame["edge_away"] = edge(1.0 - log_frame["p_home"], log_frame["away_dec"])
            # Show the most recent day logged before its games, and the most
            # recent replayed day, when the log has them.
            predictions = []
            for before in (True, False):
                part = log_frame[log_frame["made_before_game"] == before]
                if date is not None:
                    part = part[part["date"] == date]
                if len(part):
                    predictions.append(part[part["date"] == part["date"].max()].sort_values("game_id"))
    except FileNotFoundError:
        pass
    hero = "assets/hero.jpg" if (out.parent / "assets" / "hero.jpg").exists() else None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(result, predictions, card, hero))
    return out


# ---------------------------------------------------------------------------
# Keeping the README's results section in step with the latest backtest
# ---------------------------------------------------------------------------

README_START = "<!-- results:start (written by `mlb backtest`, do not edit by hand) -->"
README_END = "<!-- results:end -->"


def readme_block(result: dict) -> str:
    """The plain-language results summary that sits at the top of the README."""
    overall, betting = result["forecast"]["overall"], result.get("betting", {})
    early = result.get("early_prices", {})
    move = early.get("line_movement", {})
    model, home = overall["model"], overall["always_home"]
    seasons = result["test_seasons"]
    out = [
        "## What the results say",
        "",
        f"Tested on **{model['games']:,} games** from {season_span(seasons)}. Every season was predicted by a "
        "model that had never seen it.",
        "",
        "| Forecaster | Games | Called correctly |",
        "|---|---:|---:|",
        f"| This model | {model['games']:,} | **{pct(model['accuracy'])}** |",
        f"| Team rating (Elo) on its own | {model['games']:,} | {pct(overall['team_rating_only']['accuracy'])} |",
        f"| Always picking the home team | {model['games']:,} | {pct(home['accuracy'])} |",
    ]
    market = overall.get("market")
    if market:
        priced = overall["model_on_priced_games"]
        gap = overall["model_minus_market_log_loss"]
        out += [
            f"| Betting market's closing price | {market['games']:,} | **{pct(market['accuracy'])}** |",
            f"| This model, on those same games | {market['games']:,} | {pct(priced['accuracy'])} |",
            "",
            "In plain words:",
            "",
            f"- **The model is real.** It beats always picking the home team by "
            f"{100 * overall['model_minus_always_home_accuracy']['diff']:.1f} points, and it cannot see a game's "
            "own result.",
        ]
        detail = result.get("player_detail", {}).get("model_minus_team_only_log_loss")
        if detail and "team_only" in overall:
            team = overall["team_only"]
            if detail["high"] < 0:
                lead = "**Looking at individual players made it better.**"
            elif detail["low"] > 0:
                lead = "**Looking at individual players made it worse.**"
            else:
                lead = "**Looking at individual players made little difference.**"
            out.append(f"- {lead} With the starting lineups, each starter's pitch quality, Statcast contact data, "
                       f"injuries, age and expected runs added, log loss went from {team['log_loss']:.4f} to "
                       f"{model['log_loss']:.4f} and accuracy from {pct(team['accuracy'])} to {pct(model['accuracy'])}. "
                       "Lower log loss is better.")
        if gap["low"] > 0:
            out.append("- **It does not beat the betting market.** The closing price is the better forecaster "
                       f"(log loss {market['log_loss']:.4f} against {priced['log_loss']:.4f}, and the gap is "
                       "larger than chance).")
        elif gap["high"] < 0:
            out.append("- **It beats the betting market's closing price** on log loss "
                       f"({priced['log_loss']:.4f} against {market['log_loss']:.4f}, and the gap is larger than chance).")
        else:
            out.append("- **It is about level with the betting market.** The gap between the two is within chance "
                       f"(log loss {priced['log_loss']:.4f} against {market['log_loss']:.4f}).")
    if betting.get("seasons"):
        runs = betting["overall"]
        blind = [runs[k]["roi"] for k in ("always_home", "always_favourite", "always_underdog") if k in runs]
        agent, rule = runs.get("grpo_agent"), runs.get("edge_rule")
        line = (f"- **Paper betting at closing prices has not made money** ({season_span(betting['seasons'])}, "
                f"{betting['games']:,} games). Blind strategies lost {pct(-max(blind))} to {pct(-min(blind))} per "
                "unit staked, which is the bookmaker's cut.")
        if rule and rule["bets"]:
            line += f" A fixed \"bet when the model sees a 3% edge\" rule returned {pct(rule['roi'])}."
        out.append(line)
        notes = betting.get("agent_training", {})
        kept = sum(1 for n in notes.values() if n.get("kept_never_bet_policy"))
        if agent is not None:
            if agent["bets"] == 0:
                out.append(f"- **The learning agent chose not to bet.** In all {len(notes)} test seasons, no betting "
                           "policy it learned was proven better than not betting, so it passed on every game. "
                           "That is the right call given the line above.")
            else:
                out.append(f"- **The learning agent bet very rarely.** It passed entirely in {kept} of {len(notes)} "
                           f"test seasons and placed {agent['bets']:,} paper bets in total "
                           f"(return per unit {pct(agent['roi'])}, far too few bets to mean anything).")
    raw = betting.get("overall", {}).get("grpo_agent_safety_off") if betting.get("seasons") else None
    if raw and raw["bets"]:
        out.append(f"- **What the agent would have bet with its safety bar off:** {raw['bets']:,} paper bets, "
                   f"returning {pct(raw['roi'])} per unit staked. That simulation is why the bar exists.")
    closing_mix = result.get("market_blend", {}).get("closing", {})
    if closing_mix.get("games"):
        gap_mix = closing_mix["mix_minus_market"]
        if gap_mix["high"] < 0:
            out.append("- **The model knows something the closing price does not.** Mixing it with the market's price "
                       f"gave better forecasts than the price alone (log loss change {gap_mix['diff']:+.4f}).")
        else:
            out.append("- **The closing price already contains what the model knows.** Mixing the model with the "
                       "market's price was no better than the price alone.")
    shop = best_price_sentence(result)
    if shop:
        out.append(f"- **Shopping for the best price matters more than anything else tested.** {shop}")
    if move.get("games"):
        line = (f"- **Early prices are where the model has something to offer.** When the model disagrees with the "
                f"opening price, the price tends to move toward the model by game time ({move['games']:,} games, "
                f"{season_span(move['seasons'])}).")
        opening_mix = result.get("market_blend", {}).get("opening", {})
        if opening_mix.get("games"):
            if opening_mix["mix_minus_market"]["high"] < 0:
                line += (" Mixed with the opening price, the model gives better forecasts than the opening price "
                         "alone, by more than chance.")
            else:
                line += " Mixed with the opening price, the model was no better than the opening price alone."
        tail = early_sentence(result)
        if tail:
            line += " " + tail
        out.append(line + " The prediction log is there to test this fairly from now on.")
    out += ["",
            "Full numbers, season by season: [reports/BACKTEST.md](reports/BACKTEST.md). "
            "The same results as a page: `reports/index.html`."]
    return "\n".join(out)


def update_readme(result: dict, path: Path = Path("README.md")) -> bool:
    """Rewrite the results section of the README, if the README has the markers for it."""
    if not path.exists():
        return False
    text = path.read_text()
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(f"{head}{README_START}\n{readme_block(result)}\n{README_END}{tail}")
    return True
