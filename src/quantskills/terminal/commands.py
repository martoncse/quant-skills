"""Command-bar grammar. Pure functions, no UI: easy to test.

Examples::

    QUOTE AAPL        CHART SPY 1Y        STRAT factor-long-short
    BT factor-long-short quantile=0.1 rebalance_days=5
    ROBUST pairs-coint entry_z=2.5      RISK hrp 252
    WATCH             DATA DEMO           HELP
    AAPL              (a bare ticker means QUOTE AAPL)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from quantskills.data.views import PERIODS
from quantskills.portfolio import ALLOCATORS


class CommandError(ValueError):
    """The text in the command bar could not be parsed."""


@dataclass(frozen=True)
class Command:
    name: str
    args: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


# name -> (usage, description)
COMMANDS: dict[str, tuple[str, str]] = {
    "WATCH": ("WATCH", "Watchlist: last price, daily change, 52w range, sparkline"),
    "QUOTE": ("QUOTE <TICKER>", "Quote detail for one ticker (or just type the ticker)"),
    "CHART": ("CHART <TICKER> [1M|3M|6M|1Y|2Y|5Y|MAX]", "Price chart with SMA 50/200"),
    "STRAT": ("STRAT [strategy]", "Strategy lab: pick a strategy, edit params, run"),
    "BT": ("BT <strategy> [key=value ...]", "Run a backtest now and show results"),
    "RESULTS": ("RESULTS", "Re-open the last backtest's results"),
    "DATA": ("DATA DEMO|CACHE", "Switch between synthetic demo data and the local cache"),
    "RISK": (
        "RISK [hrp|min_var|inverse_vol|equal|eigen] [lookback]",
        "Portfolio weights, VaR/CVaR and correlation heatmap",
    ),
    "ROBUST": (
        "ROBUST [strategy] [key=value ...]",
        "Robustness: DSR, PBO, noise/delay/sensitivity, regimes, verdict",
    ),
    "DCF": (
        "DCF [case.json]",
        "Valuation: DCF, scenarios, sensitivity, comps (default: a fictional example)",
    ),
    "THEME": ("THEME [dark|amber|light]", "Switch the colour theme (no argument: next one)"),
    "HELP": ("HELP", "This list"),
    "QUIT": ("QUIT", "Exit the terminal"),
}

# How HELP groups the commands. Every command appears in exactly one group.
COMMAND_GROUPS: dict[str, tuple[str, ...]] = {
    "Market data": ("WATCH", "QUOTE", "CHART", "DATA"),
    "Strategies": ("STRAT", "BT", "RESULTS"),
    "Risk and robustness": ("RISK", "ROBUST"),
    "Valuation": ("DCF",),
    "Session": ("THEME", "HELP", "QUIT"),
}
THEME_NAMES = ("dark", "amber", "light")

ALIASES = {"W": "WATCH", "Q": "QUOTE", "C": "CHART", "H": "HELP", "?": "HELP", "EXIT": "QUIT"}


def parse_value(text: str) -> Any:
    """Parse a param value: Python literal if possible (0.1, 5, ('a','b')), else string."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_params(tokens: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise CommandError(f"expected key=value, got {tok!r}")
        key, value = tok.split("=", 1)
        if not key:
            raise CommandError(f"missing key in {tok!r}")
        params[key] = parse_value(value)
    return params


def _parse_risk(rest: list[str]) -> Command:
    if len(rest) > 2:
        raise CommandError("usage: " + COMMANDS["RISK"][0])
    method = rest[0].lower() if rest else "hrp"
    if method not in ALLOCATORS:
        raise CommandError(f"unknown method {rest[0]!r}; choose from {', '.join(ALLOCATORS)}")
    lookback = rest[1] if len(rest) == 2 else "252"
    if not lookback.isdigit() or int(lookback) < 20:
        raise CommandError("lookback must be a whole number of days >= 20")
    return Command("RISK", (method, lookback))


def parse_command(text: str) -> Command:
    tokens = text.strip().split()
    if not tokens:
        raise CommandError("empty command; type HELP")
    head = tokens[0].upper()
    name = ALIASES.get(head, head)
    rest = tokens[1:]

    if name not in COMMANDS:
        if not rest and head.replace(".", "").replace("-", "").replace("^", "").isalnum():
            return Command("QUOTE", (head,))
        raise CommandError(f"unknown command {tokens[0]!r}; type HELP")

    if name == "RISK":
        return _parse_risk(rest)
    if name == "DCF":
        if len(rest) > 1:
            raise CommandError("usage: " + COMMANDS["DCF"][0])
        return Command(name, tuple(rest))  # a path: keep its case
    if name == "THEME":
        if len(rest) > 1 or (rest and rest[0].lower() not in THEME_NAMES):
            raise CommandError(f"usage: THEME [{'|'.join(THEME_NAMES)}]")
        return Command(name, tuple(f"qs-{r.lower()}" for r in rest))
    if name in {"WATCH", "HELP", "QUIT", "RESULTS"}:
        if rest:
            raise CommandError(f"{name} takes no arguments")
        return Command(name)
    if name == "QUOTE":
        if len(rest) != 1:
            raise CommandError("usage: " + COMMANDS["QUOTE"][0])
        return Command(name, (rest[0].upper(),))
    if name == "CHART":
        if len(rest) not in (1, 2):
            raise CommandError("usage: " + COMMANDS["CHART"][0])
        period = rest[1].upper() if len(rest) == 2 else "1Y"
        if period not in PERIODS:
            raise CommandError(f"unknown period {rest[1]!r}; choose from {', '.join(PERIODS)}")
        return Command(name, (rest[0].upper(), period))
    if name == "STRAT":
        if len(rest) > 1:
            raise CommandError("usage: " + COMMANDS["STRAT"][0])
        return Command(name, tuple(t.lower() for t in rest))
    if name == "ROBUST" and not rest:
        return Command(name)  # the app picks the last-run strategy
    if name in {"BT", "ROBUST"}:
        if not rest:
            raise CommandError("usage: " + COMMANDS[name][0])
        return Command(name, (rest[0].lower(),), parse_params(rest[1:]))
    if name == "DATA":
        if len(rest) != 1 or rest[0].upper() not in {"DEMO", "CACHE"}:
            raise CommandError("usage: " + COMMANDS["DATA"][0])
        return Command(name, (rest[0].upper(),))
    raise CommandError(f"unhandled command {name}")  # pragma: no cover
