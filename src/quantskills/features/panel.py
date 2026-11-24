"""Feature and label panels for cross-sectional ML models.

A *panel* is long format: one row per ``(date, ticker)``. Features are built from
the factor functions (past data only) and z-scored across tickers each day, so a
model learns *relative* attractiveness, which is what a long-short book trades.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from quantskills.features.factors import (
    forward_returns,
    low_volatility,
    momentum,
    reversal,
    zscore_cross_section,
)
from quantskills.features.indicators import sma

ML_FEATURES: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "mom_12_1": lambda c: momentum(c, 252, 21),
    "mom_6_1": lambda c: momentum(c, 126, 21),
    "rev_5": lambda c: reversal(c, 5),
    "rev_21": lambda c: reversal(c, 21),
    "lowvol_21": lambda c: low_volatility(c, 21),
    "lowvol_63": lambda c: low_volatility(c, 63),
    "dist_sma50": lambda c: c / sma(c, 50) - 1.0,
}


def to_panel(wide: pd.DataFrame, name: str = "value") -> pd.Series:
    """Wide ``date x ticker`` -> long Series indexed ``(date, ticker)`` (NaNs kept)."""
    s = wide.stack(future_stack=True)
    s.index = s.index.set_names(["date", "ticker"])
    return s.rename(name)


def from_panel(s: pd.Series) -> pd.DataFrame:
    """Long ``(date, ticker)`` Series -> wide frame."""
    return s.unstack("ticker")


def feature_panel(close: pd.DataFrame, names: Iterable[str] | None = None) -> pd.DataFrame:
    """Cross-sectionally z-scored features, long format, one column per feature."""
    names = list(names) if names is not None else list(ML_FEATURES)
    unknown = set(names) - set(ML_FEATURES)
    if unknown:
        raise KeyError(f"unknown features: {sorted(unknown)}")
    cols = {n: to_panel(zscore_cross_section(ML_FEATURES[n](close)), n) for n in names}
    return pd.DataFrame(cols)


def label_panel(close: pd.DataFrame, horizon: int = 21, kind: str = "rank") -> pd.Series:
    """LABEL: forward return over ``horizon`` days, long format.

    ``kind``: ``"raw"`` (the return), ``"rank"`` (cross-sectional percentile minus
    0.5, robust to outliers), or ``"up"`` (1 if above the cross-sectional median, else 0).
    The last ``horizon`` dates are NaN: their outcome isn't known yet.
    """
    fwd = forward_returns(close, horizon)
    if kind == "raw":
        lab = fwd
    elif kind == "rank":
        lab = fwd.rank(axis=1, pct=True) - 0.5
    elif kind == "up":
        med = fwd.median(axis=1)
        lab = fwd.gt(med, axis=0).astype(float).where(fwd.notna())
    else:
        raise ValueError("kind must be 'raw', 'rank' or 'up'")
    return to_panel(lab, "label")
