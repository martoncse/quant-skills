"""Portfolio risk: VaR, CVaR (expected shortfall) and risk contributions.

Sign convention: VaR and CVaR are reported as **positive loss fractions** over a
one-day horizon. ``var=0.021`` at 95% means "on 1 day in 20 we expect to lose
at least 2.1%", *if the past is a fair guide*, which it often isn't in a crisis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantskills.eval.metrics import TRADING_DAYS, max_drawdown


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily returns of a portfolio rebalanced to constant ``weights`` each day."""
    w = weights.reindex(returns.columns).fillna(0.0)
    return returns.fillna(0.0) @ w


def historical_var(r: pd.Series, level: float = 0.95) -> float:
    """Empirical quantile loss: -q_{1-level}(r)."""
    _check(level)
    return float(-np.quantile(r.dropna(), 1.0 - level))


def historical_cvar(r: pd.Series, level: float = 0.95) -> float:
    """Mean loss on days at or beyond the VaR threshold (expected shortfall)."""
    _check(level)
    x = r.dropna().to_numpy()
    cut = np.quantile(x, 1.0 - level)
    return float(-x[x <= cut].mean())


def parametric_var(r: pd.Series, level: float = 0.95) -> float:
    """Gaussian VaR: -(mu + z_{1-level} * sigma)."""
    _check(level)
    x = r.dropna()
    return float(-(x.mean() + stats.norm.ppf(1.0 - level) * x.std()))


def cornish_fisher_var(r: pd.Series, level: float = 0.95) -> float:
    """VaR with the Gaussian quantile adjusted for skew and excess kurtosis."""
    _check(level)
    x = r.dropna()
    z = stats.norm.ppf(1.0 - level)
    s, k = stats.skew(x), stats.kurtosis(x)  # k is excess kurtosis
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(x.mean() + z_cf * x.std()))


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Fraction of portfolio variance from each asset: w_i (S w)_i / (w' S w). Sums to 1."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    s = cov.to_numpy()
    total = float(w @ s @ w)
    rc = w * (s @ w) / total if total > 0 else np.zeros_like(w)
    return pd.Series(rc, index=cov.index)


def risk_report(returns: pd.DataFrame, weights: pd.Series, levels=(0.95, 0.99)) -> dict[str, float]:
    """Headline risk numbers for a constant-weight portfolio over ``returns``."""
    pr = portfolio_returns(returns, weights)
    out: dict[str, float] = {
        "ann_vol": float(pr.std() * np.sqrt(TRADING_DAYS)),
        "max_drawdown": max_drawdown(pr),
        "n_days": float(len(pr)),
    }
    for lv in levels:
        tag = f"{round(lv * 100)}"
        out[f"var_hist_{tag}"] = historical_var(pr, lv)
        out[f"cvar_hist_{tag}"] = historical_cvar(pr, lv)
        out[f"var_gauss_{tag}"] = parametric_var(pr, lv)
        out[f"var_cf_{tag}"] = cornish_fisher_var(pr, lv)
    return out


def _check(level: float) -> None:
    if not 0.5 < level < 1.0:
        raise ValueError("level must be in (0.5, 1), e.g. 0.95")
