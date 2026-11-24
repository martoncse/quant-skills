"""Pick an allocator by name. Shared by the RISK screen and strategy #7."""

from __future__ import annotations

import pandas as pd

from quantskills.portfolio.covariance import ledoit_wolf_cov, sample_cov
from quantskills.portfolio.hrp import hrp_weights
from quantskills.portfolio.optimize import equal_weight, inverse_volatility, min_variance
from quantskills.portfolio.pca import eigenportfolio

ALLOCATORS = ("equal", "inverse_vol", "min_var", "hrp", "eigen")


def allocate(
    returns: pd.DataFrame,
    method: str = "hrp",
    shrink: bool = True,
    max_weight: float = 1.0,
    eigen_k: int = 0,
) -> pd.Series:
    """Long-only weights (sum 1) from a window of past returns, except ``eigen``,
    whose weights have unit gross and may be long-short for ``eigen_k > 0``.

    Only columns with a full window are used; others get weight 0.
    """
    r = returns.dropna(axis=1, how="any")
    if r.shape[1] == 0:
        raise ValueError("no asset has a complete return window")
    if method == "equal":
        w = equal_weight(r.columns)
    elif method == "eigen":
        w = eigenportfolio(r, eigen_k)
    else:
        cov = ledoit_wolf_cov(r) if shrink else sample_cov(r)
        if method == "inverse_vol":
            w = inverse_volatility(cov)
        elif method == "min_var":
            w = min_variance(cov, max_weight=max(max_weight, 1.0 / len(cov)))
        elif method == "hrp":
            w = hrp_weights(cov)
        else:
            raise ValueError(f"unknown method {method!r}; choose from {ALLOCATORS}")
    return w.reindex(returns.columns).fillna(0.0)
