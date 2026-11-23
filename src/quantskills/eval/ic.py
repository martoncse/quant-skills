"""Factor evaluation: information coefficient and quantile spreads.

Inputs are wide ``date x ticker`` frames: a factor (known at *t*) and forward
returns (label from *t* to *t+h*, see ``features.forward_returns``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
    min_obs: int = 5,
) -> pd.Series:
    """Per-date cross-sectional correlation between factor and forward returns."""
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    f, y = factor.align(fwd_returns, join="inner")
    mask = f.notna() & y.notna()
    f, y = f.where(mask), y.where(mask)
    if method == "spearman":
        f, y = f.rank(axis=1), y.rank(axis=1)
    fd = f.sub(f.mean(axis=1), axis=0)
    yd = y.sub(y.mean(axis=1), axis=0)
    num = (fd * yd).sum(axis=1)
    den = np.sqrt((fd**2).sum(axis=1) * (yd**2).sum(axis=1))
    ic = num / den.replace(0.0, np.nan)
    ic[mask.sum(axis=1) < min_obs] = np.nan
    ic.name = "ic"
    return ic.dropna()


def ic_summary(ic: pd.Series, horizon: int = 1) -> dict[str, float]:
    """Mean IC, IC std, IR and a t-stat.

    With ``horizon > 1`` consecutive IC values overlap; the t-stat uses only
    every ``horizon``-th observation so it is not overstated.
    """
    ic = ic.dropna()
    sampled = ic.iloc[::horizon] if horizon > 1 else ic
    mean, sd, n = float(ic.mean()), float(ic.std(ddof=1)), len(sampled)
    sd_s = float(sampled.std(ddof=1))
    return {
        "mean_ic": mean,
        "ic_std": sd,
        "ic_ir": mean / sd if sd > 0 else float("nan"),
        "t_stat": float(sampled.mean() / sd_s * np.sqrt(n)) if sd_s > 0 else float("nan"),
        "hit_rate": float((ic > 0).mean()),
        "n_obs": float(n),
    }


def quantile_returns(
    factor: pd.DataFrame, fwd_returns: pd.DataFrame, n_quantiles: int = 5
) -> pd.Series:
    """Mean forward return per factor quantile (1 = lowest factor values)."""
    f, y = factor.align(fwd_returns, join="inner")
    mask = f.notna() & y.notna()
    ranks = f.where(mask).rank(axis=1, pct=True)
    buckets = np.ceil(ranks * n_quantiles).clip(1, n_quantiles)
    long = pd.DataFrame({"q": buckets.stack(), "r": y.where(mask).stack()}).dropna()
    per_date = long.groupby([long.index.get_level_values(0), "q"])["r"].mean()
    out = per_date.groupby(level="q").mean()
    out.index = out.index.astype(int)
    out.index.name = "quantile"
    return out
