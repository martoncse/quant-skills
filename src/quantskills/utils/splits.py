"""Time-ordered splits. Never shuffle time series."""

from __future__ import annotations

import pandas as pd


def holdout_date(dates: pd.Index, test_fraction: float = 0.3) -> pd.Timestamp:
    """Last training date such that the final ``test_fraction`` of dates is held out."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    unique = pd.DatetimeIndex(dates).unique().sort_values()
    cut = int(len(unique) * (1.0 - test_fraction)) - 1
    return pd.Timestamp(unique[max(cut, 0)])
