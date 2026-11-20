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


def season_span(seasons: list) -> str:
    """'season 2026' or 'seasons 2024 to 2026'."""
    return f"season {seasons[0]}" if seasons[0] == seasons[-1] else f"seasons {seasons[0]} to {seasons[-1]}"


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
    for key, title in (("betting", "Paper betting at closing prices"),
                       ("early_prices", "Paper betting at opening prices")):
        betting = result.get(key, {})
        if not betting.get("seasons"):
            continue
        lines += ["", f"{title}, {season_span(betting['seasons'])} "
                      f"({betting['games']:,} games):",
                  f"{'':<32}{'Bets':>7}{'Won':>8}{'Return/unit':>13}{'Vs close':>10}"]
        for name_key, name in STRATEGY_NAMES.items():
            run = betting["overall"].get(name_key)
            if run:
                lines.append(f"{name:<32}{run['bets']:>7}{pct(run['win_rate']):>8}{pct(run['roi']):>13}"
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
        out += ["", "### The agent, season by season", "",
                "\"Kept never-bet\" means that after training, no version of the agent was proven better than "
                "not betting on the held-back season, so it passed on every game.",
                "",
                "| Season | Trained on | Checked on | Result of training | Bets | Return per unit | Edge rule return per unit |",
                "|---|---|---|---|---:|---:|---:|"]
        out += _agent_rows(betting)
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
            "One caution. The model is given the pitcher who actually started. When an opening price was set "
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


def _agent_rows(betting: dict) -> list[str]:
    rows = []
    for season, block in betting["by_season"].items():
        agent, rule = block.get("grpo_agent"), block.get("edge_rule")
        note = betting["agent_training"].get(season) or betting["agent_training"].get(str(season)) or {}
        trained = note.get("train_seasons") or []
        span = f"{trained[0]} to {trained[-1]}" if trained else ""
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
        html.append(f'<tr><td>{season}</td><td class="n">{block["model"]["games"]:,}</td>'
                    f'<td class="n">{pct(block["always_home"]["accuracy"])}</td><td class="n">{pct(block["model"]["accuracy"])}</td>'
                    f'<td class="n">{pct(mk["accuracy"]) if mk else "no odds"}</td>'
                    f'<td class="n">{block["model"]["log_loss"]:.4f}</td><td class="n">{f"{mk["log_loss"]:.4f}" if mk else ""}</td></tr>')
    html.append("</tbody></table></div></details></div>")
    html.append('<div class="card scroll"><table><thead><tr><th>Forecaster</th><th class="n">Correct</th>'
                '<th class="n">Log loss</th><th class="n">Calibration error</th></tr></thead><tbody>')
    for key, name in FORECAST_NAMES.items():
        s = overall[key]
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
                    "not support betting at these prices.</p></section>")

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
        html.append("<p class='note'>One caution: the model is given the pitcher who actually started. An opening price "
                    "set before a late pitching change could not have known that, which flatters these numbers by an "
                    "unknown amount. The prediction log measures this fairly from now on.</p></section>")

    if predictions is not None and len(predictions):
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
             "professional bettors, and the bookmaker takes a cut of about 4% on top.</p></section>",
             "</main><footer><div class='wrap'>",
             f"<p>Generated {_esc(generated or dt.date.today().isoformat())} by <code>mlb page</code>. Paper trading only.</p>",
             "<p>The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
             "Interested parties may contact Retrosheet at www.retrosheet.org. Current-season results and schedules come "
             "from MLB's public feed, and betting prices from ESPN's public feed.</p>",
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
            day = log_frame["date"].max() if date is None else date
            shown = log_frame[log_frame["date"] == day].copy()
            if len(shown):
                from .betting import edge
                shown["edge_home"] = edge(shown["p_home"], shown["home_dec"])
                shown["edge_away"] = edge(1.0 - shown["p_home"], shown["away_dec"])
                predictions = shown.sort_values("game_id")
    except FileNotFoundError:
        pass
    hero = "assets/hero.jpg" if (out.parent / "assets" / "hero.jpg").exists() else None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(result, predictions, card, hero))
    return out
