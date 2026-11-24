"""Regime-shift tests: how does a strategy do in market states it may not have seen?

Regimes are labelled from an equal-weight index of the universe using **past data
only**, so the labels themselves are not a source of look-ahead:

* ``"vol"``: trailing ``window``-day volatility above / below its own *expanding*
  median (high vol / low vol).
* ``"trend"``: index level above / below its trailing 200-day average (up / down).

``regime_table`` lags the labels one day: the regime known at the close of *t-1*
is matched with the return earned from *t-1* to *t*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantskills.eval.metrics import (
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)

REGIME_KINDS = ("vol", "trend")


def market_returns(close: pd.DataFrame) -> pd.Series:
    """Equal-weight daily return of all tickers with a price on both days."""
    return close.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)


def regime_labels(
    close: pd.DataFrame, kind: str = "vol", window: int = 63, trend_window: int = 200
) -> pd.Series:
    """Label each date with a regime using data up to that date. NaN during warm-up."""
    mkt = market_returns(close)
    if kind == "vol":
        rv = mkt.rolling(window).std()
        threshold = rv.expanding(min_periods=window).median()
        lab = np.where(rv > threshold, "high vol", "low vol")
        ok = rv.notna() & threshold.notna()
    elif kind == "trend":
        level = (1.0 + mkt).cumprod()
        avg = level.rolling(trend_window).mean()
        lab = np.where(level > avg, "up trend", "down trend")
        ok = avg.notna()
    else:
        raise ValueError(f"kind must be one of {REGIME_KINDS}")
    return pd.Series(lab, index=close.index, name=kind).where(ok)


def regime_table(returns: pd.Series, labels: pd.Series, min_days: int = 63) -> pd.DataFrame:
    """Per-regime days, share, annualised return / vol, Sharpe, max drawdown.

    Regimes with fewer than ``min_days`` days get NaN statistics (too few to judge).
    """
    lagged = labels.shift(1).reindex(returns.index)
    df = pd.DataFrame({"r": returns, "regime": lagged}).dropna()
    rows = {}
    for name, g in df.groupby("regime"):
        r = g["r"]
        enough = len(r) >= min_days
        rows[name] = {
            "days": len(r),
            "share": len(r) / len(df),
            "ann_return": annualized_return(r, TRADING_DAYS) if enough else np.nan,
            "ann_vol": annualized_volatility(r) if enough else np.nan,
            "sharpe": sharpe_ratio(r) if enough else np.nan,
            "max_drawdown": max_drawdown(r) if enough else np.nan,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "regime"
    return out


def regime_report(
    returns: pd.Series, close: pd.DataFrame, kinds: tuple[str, ...] = REGIME_KINDS
) -> pd.DataFrame:
    """Stack :func:`regime_table` for several regime definitions (MultiIndex kind, regime)."""
    parts = {k: regime_table(returns, regime_labels(close, k)) for k in kinds}
    return pd.concat(parts, names=["kind", "regime"])
