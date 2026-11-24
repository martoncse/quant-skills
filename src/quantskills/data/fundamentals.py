"""Starter valuation cases from Yahoo Finance fundamentals (network; cached as JSON).

``starter_case(ticker)`` fills the *base year* and balance-sheet fields from the latest
annual statements and leaves every **forecast assumption** to you: it copies the last
margin flat and uses a cautious growth path, and writes no thesis. Treat the output as a
first draft to edit, never as a finished case. Yahoo's field names change over time;
missing fields raise a clear error naming them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

Downloader = Callable[[str], dict[str, Any]]


def _download_yfinance(ticker: str) -> dict[str, Any]:  # pragma: no cover - network
    import yfinance as yf

    t = yf.Ticker(ticker)
    return {
        "info": dict(t.info),
        "income": t.financials,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
    }


def _row(df: pd.DataFrame, names: tuple[str, ...], what: str) -> pd.Series:
    for n in names:
        if n in df.index:
            return df.loc[n].dropna().sort_index()
    raise KeyError(f"no {what} row (tried {', '.join(names)})")


def starter_case(
    ticker: str,
    downloader: Downloader | None = None,
    cache_dir: str | Path = "data/fundamentals",
    years: int = 5,
) -> dict[str, Any]:
    """A case dict for :class:`quantskills.fundamental.Case`, base year from reported data."""
    raw = (downloader or _download_yfinance)(ticker)
    info, inc, cf = raw["info"], raw["income"], raw["cashflow"]
    revenue = _row(inc, ("Total Revenue", "Operating Revenue"), "revenue")
    ebit = _row(inc, ("EBIT", "Operating Income"), "EBIT")
    pretax = _row(inc, ("Pretax Income",), "pre-tax income")
    tax = _row(inc, ("Tax Provision", "Income Tax Expense"), "tax")
    capex = _row(cf, ("Capital Expenditure",), "capex").abs()
    da = _row(cf, ("Depreciation And Amortization", "Depreciation"), "D&A")
    rev = float(revenue.iloc[-1])
    tax_rate = (
        float(np.clip(tax.iloc[-1] / pretax.iloc[-1], 0.0, 0.35)) if pretax.iloc[-1] > 0 else 0.21
    )
    growth_hist = revenue.pct_change().dropna()
    g0 = float(np.clip(growth_hist.tail(3).mean(), -0.05, 0.15)) if len(growth_hist) else 0.03
    margin = float(ebit.iloc[-1] / rev)
    debt = float(info.get("totalDebt") or 0.0)
    cash = float(info.get("totalCash") or 0.0)
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares = info.get("sharesOutstanding")
    if not price or not shares:
        raise KeyError("price or sharesOutstanding missing from Yahoo info")
    case = {
        "company": {
            "name": info.get("longName") or ticker,
            "ticker": ticker.upper(),
            "price": float(price),
            "shares_out": float(shares) / 1e6,
            "net_debt": (debt - cash) / 1e6,
            "currency": info.get("currency", "USD"),
            "as_of": date.today().isoformat(),
            "fictional": False,
            "source": "Yahoo Finance via yfinance (unaudited aggregation; check the filings)",
        },
        "base": {
            "revenue": rev / 1e6,
            "ebit_margin": margin,
            "tax_rate": tax_rate,
            "da_pct_revenue": float(da.iloc[-1]) / rev,
            "capex_pct_revenue": float(capex.iloc[-1]) / rev,
            "nwc_pct_revenue": 0.10,  # placeholder: set from the balance sheet
        },
        "forecast": {
            # Starts at the recent growth rate and fades linearly towards 3%: an assumption.
            "revenue_growth": list(np.linspace(g0, 0.03, years).round(4)),
            "ebit_margin": [round(margin, 4)] * years,
        },
        "discount": {
            "risk_free": 0.042,
            "equity_risk_premium": 0.05,
            "beta": float(info.get("beta") or 1.0),
            "pre_tax_cost_of_debt": 0.06,
            "debt_weight": float(np.clip(debt / (debt + float(price) * float(shares)), 0, 0.8))
            if debt
            else 0.0,
        },
        "terminal": {"growth": 0.025},
        "peers": [],
        "scenarios": {},
        "thesis": [],
        "risks": [],
        "catalysts": [],
    }
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{ticker.upper()}.json").write_text(
        json.dumps(case, indent=2), encoding="utf-8", newline="\n"
    )
    return case
