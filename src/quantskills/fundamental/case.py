"""A valuation *case*: everything a DCF and a one-page pitch need, in one JSON file.

Money amounts are in millions of the company's currency; per-share values are in
currency units. Rates are decimals (0.08 = 8%).

The written parts (thesis, risks, catalysts) are **the analyst's own words**: the
pitch generator lays them out but never writes them. Set ``"fictional": true``
for illustrative cases; the pitch then says so in its title.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Company:
    name: str
    ticker: str
    price: float  # current share price
    shares_out: float  # diluted shares, millions
    net_debt: float  # debt - cash, millions (negative = net cash)
    currency: str = "USD"
    as_of: str = ""  # date of the price and balance sheet
    fictional: bool = False
    source: str = ""  # where the numbers came from


@dataclass
class Base:
    """Last reported fiscal year."""

    revenue: float
    ebit_margin: float
    tax_rate: float
    da_pct_revenue: float  # depreciation & amortisation
    capex_pct_revenue: float
    nwc_pct_revenue: float  # net working capital as a share of revenue


@dataclass
class Forecast:
    revenue_growth: list[float]  # one per forecast year
    ebit_margin: list[float]  # one per forecast year
    capex_pct_revenue: float | None = None  # default: Base
    da_pct_revenue: float | None = None
    nwc_pct_revenue: float | None = None

    @property
    def years(self) -> int:
        return len(self.revenue_growth)


@dataclass
class Discount:
    risk_free: float
    equity_risk_premium: float
    beta: float
    pre_tax_cost_of_debt: float
    debt_weight: float  # target D / (D + E)
    wacc_override: float | None = None
    mid_year: bool = True  # cash flows arrive mid-year on average


@dataclass
class Terminal:
    growth: float  # perpetual growth after the forecast
    exit_ev_ebitda: float | None = None  # optional cross-check


@dataclass
class Peer:
    name: str
    ev_ebitda: float | None = None
    pe: float | None = None


@dataclass
class Scenario:
    """Shifts applied to the base case (added to every forecast year)."""

    growth_shift: float = 0.0
    margin_shift: float = 0.0
    wacc_shift: float = 0.0
    terminal_growth: float | None = None


@dataclass
class Case:
    company: Company
    base: Base
    forecast: Forecast
    discount: Discount
    terminal: Terminal
    peers: list[Peer] = field(default_factory=list)
    scenarios: dict[str, Scenario] = field(default_factory=dict)
    thesis: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        try:
            return cls(
                company=Company(**d["company"]),
                base=Base(**d["base"]),
                forecast=Forecast(**d["forecast"]),
                discount=Discount(**d["discount"]),
                terminal=Terminal(**d["terminal"]),
                peers=[Peer(**p) for p in d.get("peers", [])],
                scenarios={k: Scenario(**v) for k, v in d.get("scenarios", {}).items()},
                thesis=list(d.get("thesis", [])),
                risks=list(d.get("risks", [])),
                catalysts=list(d.get("catalysts", [])),
            )
        except KeyError as exc:
            raise ValueError(f"case is missing section {exc}") from None
        except TypeError as exc:
            raise ValueError(f"bad field in case: {exc}") from None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate(case: Case) -> None:
    """Raise ``ValueError`` on inputs that would make the DCF meaningless."""
    f, c, b = case.forecast, case.company, case.base
    if f.years < 1 or len(f.ebit_margin) != f.years:
        raise ValueError("forecast needs one revenue_growth and one ebit_margin per year")
    if c.shares_out <= 0 or c.price <= 0 or b.revenue <= 0:
        raise ValueError("price, shares_out and revenue must be positive")
    if not 0 <= b.tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    if not 0 <= case.discount.debt_weight < 1:
        raise ValueError("debt_weight must be in [0, 1)")
    if any(g <= -1 for g in f.revenue_growth):
        raise ValueError("revenue growth must be > -100%")
    if case.terminal.growth > 0.05:
        raise ValueError(
            "terminal growth above 5% a year forever exceeds long-run nominal GDP growth"
        )


def load_case(path: str | Path) -> Case:
    return Case.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_case(case: Case, path: str | Path) -> None:
    text = json.dumps(case.to_dict(), indent=2)
    Path(path).write_text(text, encoding="utf-8", newline="\n")  # LF on every OS


def example_case() -> Case:
    """The bundled, clearly fictional example (``Example Industrial Co.``)."""
    from importlib.resources import files

    text = files("quantskills.fundamental").joinpath("examples/example_industrial.json")
    return Case.from_dict(json.loads(text.read_text(encoding="utf-8")))
