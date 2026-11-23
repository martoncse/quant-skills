"""Chart indicators. Trailing windows only (row t uses rows <= t)."""

from __future__ import annotations

import pandas as pd


def sma(close: pd.DataFrame | pd.Series, window: int = 50):
    """Simple moving average over the trailing ``window`` rows (NaN until full)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.DataFrame | pd.Series, span: int = 20):
    """Exponential moving average, alpha = 2 / (span + 1), trailing only."""
    if span < 1:
        raise ValueError("span must be >= 1")
    return close.ewm(span=span, adjust=False, min_periods=span).mean()
