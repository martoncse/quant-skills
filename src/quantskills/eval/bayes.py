"""Bayesian Sharpe ratio: a posterior distribution instead of a point estimate.

Model: daily returns r_t ~ Normal(mu, sigma^2), i.i.d., with the standard
non-informative prior p(mu, sigma^2) proportional to 1/sigma^2. The posterior is then
known in closed form (normal / scaled-inverse-chi-square), so we sample it
exactly with numpy: no MCMC library needed. (PyMC would give the same answer
for this model; it earns its keep for models without a closed form.)

    sigma^2 | r  ~  (n-1) s^2 / chi^2_{n-1}
    mu | sigma^2, r  ~  Normal(rbar, sigma^2 / n)

Each draw gives one Sharpe value mu / sigma * sqrt(252). Fat tails and serial
correlation are ignored, so treat the spread as a lower bound on the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantskills.eval.metrics import TRADING_DAYS


@dataclass
class BayesianSharpe:
    mean: float
    median: float
    ci_low: float
    ci_high: float
    prob_positive: float
    draws: np.ndarray

    def __repr__(self) -> str:
        return (
            f"BayesianSharpe(mean={self.mean:.2f}, 90% CI=[{self.ci_low:.2f}, {self.ci_high:.2f}],"
            f" P(SR>0)={self.prob_positive:.3f})"
        )


def bayesian_sharpe(
    returns: pd.Series,
    n_draws: int = 20_000,
    credible: float = 0.90,
    seed: int = 0,
    periods: int = TRADING_DAYS,
) -> BayesianSharpe:
    """Posterior of the annualised Sharpe ratio of daily ``returns``."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 3:
        raise ValueError("need at least 3 returns")
    rng = np.random.default_rng(seed)
    s2 = r.var(ddof=1)
    sigma2 = (n - 1) * s2 / rng.chisquare(n - 1, n_draws)
    mu = rng.normal(r.mean(), np.sqrt(sigma2 / n))
    sr = mu / np.sqrt(sigma2) * np.sqrt(periods)
    tail = (1.0 - credible) / 2.0
    lo, hi = np.quantile(sr, [tail, 1.0 - tail])
    return BayesianSharpe(
        mean=float(sr.mean()),
        median=float(np.median(sr)),
        ci_low=float(lo),
        ci_high=float(hi),
        prob_positive=float((sr > 0).mean()),
        draws=sr,
    )
