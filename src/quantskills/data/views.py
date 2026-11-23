"""Read-only views of a price panel: quote tables and period windows.

Used by the terminal; kept here so the UI holds no calculations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide

PERIODS: dict[str, int | None] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "5Y": 1260,
    "MAX": None,
}


def period_window(obj: pd.DataFrame | pd.Series, period: str = "1Y"):
    """Last ``period`` of rows (trading-day approximation). ``MAX`` returns everything."""
    key = period.upper()
    if key not in PERIODS:
        raise ValueError(f"unknown period {period!r}; choose from {list(PERIODS)}")
    n = PERIODS[key]
    return obj if n is None else obj.iloc[-n:]


def quote_table(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: last close, daily change, 52-week range, 3M vol, as-of date.

    Each ticker uses its own last valid close, so a stale ticker shows its own date.
    """
    close = to_wide(prices, "close")
    rows = {}
    for ticker in close.columns:
        s = close[ticker].dropna()
        if s.empty:
            continue
        last = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) > 1 else np.nan
        year = s.iloc[-252:]
        rets = s.pct_change(fill_method=None).iloc[-63:]
        rows[ticker] = {
            "last": last,
            "change": last - prev,
            "change_pct": last / prev - 1.0 if prev else np.nan,
            "high_52w": float(year.max()),
            "low_52w": float(year.min()),
            "vol_3m": float(rets.std() * np.sqrt(252)) if len(rets) > 2 else np.nan,
            "as_of": s.index[-1],
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "ticker"
    return out


def quote_detail(prices: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """``quote_table`` row for one ticker plus period returns. Raises ``KeyError``."""
    table = quote_table(prices)
    if ticker not in table.index:
        raise KeyError(f"unknown ticker {ticker!r}")
    detail: dict[str, Any] = table.loc[ticker].to_dict()
    s = to_wide(prices, "close")[ticker].dropna()
    for label in ("1M", "3M", "1Y"):
        w = period_window(s, label)
        detail[f"ret_{label}"] = float(w.iloc[-1] / w.iloc[0] - 1.0) if len(w) > 1 else np.nan
    year_start = s[s.index.year == s.index[-1].year]
    prior = s[s.index < year_start.index[0]]
    base = prior.iloc[-1] if len(prior) else year_start.iloc[0]
    detail["ret_YTD"] = float(s.iloc[-1] / base - 1.0)
    detail["n_days"] = len(s)
    detail["first_date"] = s.index[0]
    return detail
