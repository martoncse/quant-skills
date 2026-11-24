"""Turn scores into target weights. Shared by every cross-sectional strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_long_short(
    score: pd.DataFrame,
    quantile: float = 0.2,
    gross: float = 1.0,
    rebalance_days: int = 21,
    min_assets: int = 10,
) -> pd.DataFrame:
    """Long the top ``quantile`` of ``score``, short the bottom, equal-weighted.

    Dollar neutral with ``gross`` total exposure. Rebalances every
    ``rebalance_days`` rows from the first date with at least ``min_assets``
    scored tickers; other rows are NaN ("hold"). Rows before that are 0.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    valid = score.notna().sum(axis=1)
    tradable = valid >= min_assets
    weights = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    if not tradable.any():
        return weights.fillna(0.0)

    first = int(np.argmax(tradable.to_numpy()))
    rebal_dates = score.index[first::rebalance_days]

    ranks = score.loc[rebal_dates].rank(axis=1, pct=True)
    longs = ranks > 1.0 - quantile
    shorts = ranks <= quantile
    n_long = longs.sum(axis=1).replace(0, np.nan)
    n_short = shorts.sum(axis=1).replace(0, np.nan)
    w = longs.astype(float).div(n_long, axis=0) - shorts.astype(float).div(n_short, axis=0)
    w = (w * gross / 2.0).fillna(0.0)
    w.loc[~tradable.loc[rebal_dates]] = 0.0
    weights.loc[rebal_dates] = w
    weights.iloc[:first] = 0.0
    return weights


def every_n_rows(target: pd.DataFrame, n: int, start: int = 0) -> pd.DataFrame:
    """Keep ``target`` on every n-th row from ``start``; NaN ("hold") elsewhere, 0 before."""
    out = pd.DataFrame(np.nan, index=target.index, columns=target.columns)
    out.iloc[start::n] = target.iloc[start::n]
    out.iloc[:start] = 0.0
    return out
