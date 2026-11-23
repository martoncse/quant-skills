"""Price-based alpha factors.

All functions take a wide ``date x ticker`` close-price frame. **Features** at
row *t* use only rows <= *t*; **labels** (``forward_returns``) are the only
thing allowed to look forward, and are shifted, never the features.

Value and quality factors need point-in-time fundamentals, which free sources
(yfinance) do not provide without look-ahead; they are deferred until a PIT
source is wired in (see ``docs/references.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def simple_returns(close: pd.DataFrame) -> pd.DataFrame:
    """r_t = P_t / P_{t-1} - 1."""
    return close.pct_change(fill_method=None)


def momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 style momentum: P_{t-skip} / P_{t-lookback} - 1 (Jegadeesh & Titman 1993).

    Skipping the most recent month avoids mixing in short-term reversal.
    """
    if skip >= lookback:
        raise ValueError("skip must be < lookback")
    return close.shift(skip) / close.shift(lookback) - 1.0


def reversal(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Short-term reversal: -(P_t / P_{t-window} - 1)."""
    return -(close / close.shift(window) - 1.0)


def low_volatility(close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Low-volatility anomaly: minus the trailing std of daily returns."""
    return -simple_returns(close).rolling(window, min_periods=window).std()


FACTORS: dict[str, Callable[..., pd.DataFrame]] = {
    "momentum": momentum,
    "reversal": reversal,
    "low_vol": low_volatility,
}


def compute_factors(
    close: pd.DataFrame, names: Iterable[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Compute the named factors (default: all in ``FACTORS``) with default params."""
    names = list(names) if names is not None else list(FACTORS)
    unknown = set(names) - set(FACTORS)
    if unknown:
        raise KeyError(f"unknown factors: {sorted(unknown)}")
    return {n: FACTORS[n](close) for n in names}


def zscore_cross_section(factor: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Per-date z-score across tickers, clipped to +/- ``clip``."""
    mu = factor.mean(axis=1)
    sd = factor.std(axis=1).replace(0.0, np.nan)
    return factor.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def rank_cross_section(factor: pd.DataFrame) -> pd.DataFrame:
    """Per-date percentile rank in (0, 1]."""
    return factor.rank(axis=1, pct=True)


def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """LABEL: return from close *t* to close *t+horizon*, stored at row *t*.

    Never use this as a feature. The last ``horizon`` rows are NaN by construction.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0
