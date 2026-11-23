"""Probabilistic and Deflated Sharpe ratios.

References:
    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".

All internal Sharpe values are *per period* (not annualised); kurtosis is the
non-excess (Pearson) kurtosis, so a normal distribution has kurtosis 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from quantskills.eval.metrics import TRADING_DAYS

EULER_GAMMA = 0.5772156649015329


def _moments(returns: pd.Series) -> tuple[float, float, float, int]:
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3:
        raise ValueError("need at least 3 observations")
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    return sr, skew, kurt, n


def _sr_std(sr: float, skew: float, kurt: float, n: int) -> float:
    """Std of the per-period Sharpe estimator under non-normal returns."""
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)
    return float(np.sqrt(max(var, 1e-16)))


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark), with ``sr_benchmark`` per period.

    PSR = Phi( (SR_hat - SR*) / sigma(SR_hat) ).
    """
    sr, skew, kurt, n = _moments(returns)
    return float(stats.norm.cdf((sr - sr_benchmark) / _sr_std(sr, skew, kurt, n)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] across ``n_trials`` independent zero-skill trials (per period).

    E[max] ~ sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


@dataclass(frozen=True)
class DeflatedSharpe:
    sharpe: float  # annualised observed Sharpe
    sharpe_threshold: float  # annualised E[max SR] under the null of no skill
    psr: float  # P(SR > 0)
    dsr: float  # P(SR > threshold): the multiple-testing-adjusted confidence
    n_trials: int


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    trial_sharpes: Sequence[float] | None = None,
    periods: int = TRADING_DAYS,
) -> DeflatedSharpe:
    """Deflated Sharpe ratio for the best of ``n_trials`` tried configurations.

    ``trial_sharpes`` are the *annualised* Sharpes of every trial; their
    variance sets the null threshold. Without them we fall back to the sampling
    variance of the Sharpe estimator itself (i.e. assume trials differ only by
    noise), which is the least conservative honest default. Always report the
    true number of trials, including the ones you discarded.
    """
    sr, skew, kurt, n = _moments(returns)
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        per = np.asarray(trial_sharpes, dtype=float) / np.sqrt(periods)
        sr_var = float(np.var(per, ddof=1))
        n_trials = max(n_trials, len(trial_sharpes))
    else:
        sr_var = _sr_std(sr, skew, kurt, n) ** 2
    threshold = expected_max_sharpe(n_trials, sr_var)
    sd = _sr_std(sr, skew, kurt, n)
    return DeflatedSharpe(
        sharpe=float(sr * np.sqrt(periods)),
        sharpe_threshold=float(threshold * np.sqrt(periods)),
        psr=float(stats.norm.cdf(sr / sd)),
        dsr=float(stats.norm.cdf((sr - threshold) / sd)),
        n_trials=n_trials,
    )
