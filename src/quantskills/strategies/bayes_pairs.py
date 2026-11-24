"""Strategy #5: Bayesian pairs trading with a Kalman-filter hedge ratio.

Hypothesis: as in #3, but the hedge ratio drifts over time. A fixed ratio from the
training window goes stale; a Bayesian update each day tracks it.

Method:
    1. Pairs are selected on the training window as in #3.
    2. State-space model, updated every day with Bayes' rule (the Kalman filter):

           log y_t = c_t + h_t * log x_t + e_t,   e_t ~ N(0, R)
           (c_t, h_t) = (c_{t-1}, h_{t-1}) + w_t,  w_t ~ N(0, delta * R * I)

       The prior is the training-window OLS fit; ``R`` is its residual variance.
       ``delta`` sets how fast the hedge ratio may drift, relative to the spread's
       noise. Too large and the filter "explains away" the spread, so z never
       reaches the entry band; too small and it is the fixed ratio of #3.
    3. The one-step prediction error divided by its predicted std is a z-score
       that already accounts for uncertainty in the hedge ratio. Trade it with the
       same bands as #3, using the posterior hedge ratio of the day.

The "Bayesian Sharpe" half of this strategy family lives in
``quantskills.eval.bayesian_sharpe``: a posterior over the Sharpe ratio instead
of a single number. It is reported for this strategy in its README.

Book reference: ML4T 3rd ed., ch. 9 (Kalman filter, rolling hedge ratios) and
ch. 16 (Sharpe inference).
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.strategies._pairs import Pair, band_positions, select_pairs, spread_weights
from quantskills.strategies.base import Strategy
from quantskills.strategies.pairs_coint import PAIR_DEFAULTS


def kalman_hedge(
    y: np.ndarray,
    x: np.ndarray,
    prior: tuple[float, float],
    obs_var: float,
    delta: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter (c_t, h_t). Returns (intercept, hedge, z) arrays; z is the standardised
    one-step prediction error. Every output at t uses observations up to t only.
    NaN inputs skip the update (the state carries over)."""
    n = len(y)
    theta = np.array(prior, dtype=float)
    W = np.eye(2) * delta * obs_var
    P = 10.0 * W  # prior: confident in the training OLS fit
    c_out, h_out, z_out = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        P = P + W
        if np.isfinite(y[t]) and np.isfinite(x[t]):
            H = np.array([1.0, x[t]])
            err = y[t] - H @ theta
            Q = float(H @ P @ H) + obs_var
            K = P @ H / Q
            theta = theta + K * err
            P = P - np.outer(K, H) @ P
            z_out[t] = err / np.sqrt(Q)
        c_out[t], h_out[t] = theta
    return c_out, h_out, z_out


class BayesianPairs(Strategy):
    """Pairs with a Kalman-filtered (Bayesian) hedge ratio and z-score bands."""

    name = "bayes-pairs"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {**PAIR_DEFAULTS, "delta": 1e-5}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        if not 0 <= p["exit_z"] < p["entry_z"] < p["stop_z"]:
            raise ValueError("need 0 <= exit_z < entry_z < stop_z")
        if not 0 < p["delta"] < 1:
            raise ValueError("delta must be in (0, 1)")
        self.pairs_: list[Pair] = []
        self.obs_var_: dict[str, float] = {}

    def fit(self, data: pd.DataFrame) -> Self:
        p = self.params
        log_close = np.log(to_wide(data, "close"))
        self.pairs_ = select_pairs(
            log_close,
            p["method"],
            p["max_pairs"],
            p["pvalue"],
            p["max_candidates"],
            p["bonferroni"],
        )
        self.obs_var_ = {}
        for q in self.pairs_:
            resid = log_close[q.y] - q.intercept - q.hedge * log_close[q.x]
            self.obs_var_[f"{q.y}/{q.x}"] = float(resid.var())
        self.is_fitted = True
        return self

    def filtered(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Per pair: a frame with columns intercept, hedge, z."""
        log_close = np.log(to_wide(data, "close"))
        out = {}
        for q in self.pairs_:
            key = f"{q.y}/{q.x}"
            c, h, z = kalman_hedge(
                log_close[q.y].to_numpy(),
                log_close[q.x].to_numpy(),
                (q.intercept, q.hedge),
                self.obs_var_[key],
                self.params["delta"],
            )
            out[key] = pd.DataFrame({"intercept": c, "hedge": h, "z": z}, index=log_close.index)
        return out

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        p = self.params
        close = to_wide(data, "close")
        legs = []
        for q, f in zip(self.pairs_, self.filtered(data).values(), strict=True):
            pos = band_positions(f["z"].to_numpy(), p["entry_z"], p["exit_z"], p["stop_z"])
            legs.append((q.y, q.x, pos, np.clip(f["hedge"].to_numpy(), 0.0, None)))
        return spread_weights(close.index, close.columns, legs, p["gross"])
