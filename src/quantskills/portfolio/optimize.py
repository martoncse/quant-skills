"""Mean-variance style allocators, solved with scipy (no cvxpy dependency).

All return a ``pd.Series`` of weights indexed by ticker that sums to 1.
``cov`` and ``mu`` must be on the same (e.g. annual) scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(tickers: pd.Index | list[str]) -> pd.Series:
    idx = pd.Index(tickers)
    return pd.Series(1.0 / len(idx), index=idx)


def inverse_volatility(cov: pd.DataFrame) -> pd.Series:
    """w_i proportional to 1 / sigma_i."""
    inv = 1.0 / np.sqrt(np.diag(cov.to_numpy()))
    return pd.Series(inv / inv.sum(), index=cov.index)


def _solve(
    objective, n: int, long_only: bool, max_weight: float, x0: np.ndarray | None = None
) -> np.ndarray:
    lo = 0.0 if long_only else -max_weight
    bounds = [(lo, max_weight)] * n
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"max_weight={max_weight} is infeasible for {n} assets")
    start = np.full(n, 1.0 / n) if x0 is None else x0
    res = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"optimiser failed: {res.message}")
    w = res.x
    if long_only:
        w = np.clip(w, 0.0, None)
    return w / w.sum()


def min_variance(cov: pd.DataFrame, long_only: bool = True, max_weight: float = 1.0) -> pd.Series:
    """argmin w' S w  s.t. sum(w) = 1 (and 0 <= w <= max_weight if long-only)."""
    s = cov.to_numpy()
    w = _solve(lambda w: w @ s @ w, len(s), long_only, max_weight)
    return pd.Series(w, index=cov.index)


def mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 1.0,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> pd.Series:
    """argmax w' mu - (lambda / 2) w' S w  s.t. sum(w) = 1 (Markowitz, 1952).

    Expected returns ``mu`` are the weakest input: small errors in them swing the
    weights a lot, which is why HRP and min-variance ignore them.
    """
    m = mu.reindex(cov.index).to_numpy()
    s = cov.to_numpy()
    w = _solve(
        lambda w: -(w @ m) + 0.5 * risk_aversion * (w @ s @ w), len(s), long_only, max_weight
    )
    return pd.Series(w, index=cov.index)


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> pd.Series:
    """Tangency portfolio: argmax (w' mu - rf) / sqrt(w' S w)."""
    m = mu.reindex(cov.index).to_numpy() - risk_free
    s = cov.to_numpy()

    def neg_sharpe(w: np.ndarray) -> float:
        return -float(w @ m) / float(np.sqrt(max(w @ s @ w, 1e-18)))

    w = _solve(neg_sharpe, len(s), long_only, max_weight)
    return pd.Series(w, index=cov.index)
