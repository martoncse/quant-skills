"""Time-ordered splits. Never shuffle time series.

Labels in this library are forward returns: the label at date *t* spans
``t .. t + horizon``. A training row whose label window overlaps the test
period leaks test information, so the splitters below **purge** those rows,
and optionally **embargo** a few rows after the test block (López de Prado,
*Advances in Financial ML*, ch. 7).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def holdout_date(dates: pd.Index, test_fraction: float = 0.3) -> pd.Timestamp:
    """Last training date such that the final ``test_fraction`` of dates is held out."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    unique = pd.DatetimeIndex(dates).unique().sort_values()
    cut = int(len(unique) * (1.0 - test_fraction)) - 1
    return pd.Timestamp(unique[max(cut, 0)])


def _unique_dates(dates: pd.Index | np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates).unique().sort_values()


def purged_kfold(
    dates: pd.Index | np.ndarray,
    n_splits: int = 5,
    horizon: int = 1,
    embargo: int = 0,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Contiguous k-fold over unique dates, with purging and embargo.

    For each fold, the test block is one contiguous run of dates. Training dates
    are everything else, minus:

    * the ``horizon`` dates just before the test block (their labels reach into it);
    * the ``embargo`` dates just after it (serial correlation leaks backwards).

    Yields ``(train_dates, test_dates)``. Pass ``horizon`` equal to the label's
    forward-return horizon.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    u = _unique_dates(dates)
    n = len(u)
    if n < 2 * n_splits:
        raise ValueError("not enough dates for this many splits")
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        keep = np.ones(n, dtype=bool)
        keep[max(lo - horizon, 0) : min(hi + embargo, n)] = False
        yield u[keep], u[lo:hi]


def walk_forward_splits(
    dates: pd.Index | np.ndarray,
    n_splits: int = 5,
    horizon: int = 1,
    min_train: int | None = None,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Expanding-window walk-forward splits: train on the past, test on the next block.

    Training stops ``horizon`` dates before each test block so no training label
    overlaps it. ``min_train`` defaults to one block.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    u = _unique_dates(dates)
    n = len(u)
    block = n // (n_splits + 1)
    start = block if min_train is None else min_train
    if block < 1 or start + horizon >= n:
        raise ValueError("not enough dates for this many splits")
    test_size = (n - start) // n_splits
    for k in range(n_splits):
        lo = start + k * test_size
        hi = n if k == n_splits - 1 else lo + test_size
        train_end = lo - horizon
        if train_end <= 0:
            continue
        yield u[:train_end], u[lo:hi]
