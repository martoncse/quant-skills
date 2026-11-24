"""Strategy #4: volatility timing with GARCH forecasts.

Hypothesis: volatility clusters and is forecastable even when returns are not.
Scaling exposure by forecast volatility (``target_vol / sigma_hat``) keeps risk
roughly constant and, historically, has improved risk-adjusted returns because
high-vol periods tend to have poor return per unit of risk (Moreira & Muir, 2017).

Method:
    1. ``fit``: estimate GARCH(1,1) on training-window daily returns of one asset
       (with an AR(1) mean if ``direction="ar1"``) using ``arch``.
    2. ``signals``: run the GARCH recursion forward with the **frozen** parameters:

           sigma^2_{t+1} = omega + alpha * eps_t^2 + beta * sigma^2_t

       so the forecast at t uses returns up to t only.
    3. Weight = min(target_vol / annualised sigma_{t+1}, max_leverage). With
       ``direction="ar1"`` the position is cut to zero when the AR(1) mean
       forecast is negative (the ARIMA half of the idea).

Book reference: ML4T 3rd ed., ch. 9 (ARIMA, GARCH). Bollerslev (1986).
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from arch import arch_model

from quantskills.data.loader import to_wide
from quantskills.eval.metrics import TRADING_DAYS
from quantskills.strategies._weights import every_n_rows
from quantskills.strategies.base import Strategy


def garch_forecast(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    mu: float,
    phi: float,
    sigma2_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step-ahead mean and variance forecasts made at each t (for t+1).

    Units follow ``returns`` (here: percent). NaN returns leave the state unchanged.
    """
    n = len(returns)
    mean_f, var_f = np.empty(n), np.empty(n)
    sigma2 = sigma2_0
    m = mu / (1.0 - phi) if abs(phi) < 1 else mu  # unconditional mean as the first forecast
    for t in range(n):
        r = returns[t]
        if np.isfinite(r):
            eps = r - m
            sigma2 = omega + alpha * eps**2 + beta * sigma2
            m = mu + phi * r
        mean_f[t], var_f[t] = m, sigma2
    return mean_f, var_f


class VolatilityTiming(Strategy):
    """Size one asset by its GARCH(1,1) volatility forecast to hit a target vol."""

    name = "vol-timing"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "ticker": "",  # empty: SPY if present, else the first ticker
            "target_vol": 0.10,
            "max_leverage": 2.0,
            "direction": "long",  # "long" or "ar1"
            "rebalance_days": 5,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["direction"] not in {"long", "ar1"}:
            raise ValueError("direction must be 'long' or 'ar1'")
        if self.params["target_vol"] <= 0 or self.params["max_leverage"] <= 0:
            raise ValueError("target_vol and max_leverage must be positive")
        self.ticker_: str = ""
        self.garch_params_: dict[str, float] = {}

    def _pick_ticker(self, close: pd.DataFrame) -> str:
        want = str(self.params["ticker"]).upper()
        if want:
            if want not in close.columns:
                raise KeyError(f"ticker {want!r} not in data")
            return want
        return "SPY" if "SPY" in close.columns else str(sorted(close.columns)[0])

    def fit(self, data: pd.DataFrame) -> Self:
        close = to_wide(data, "close")
        self.ticker_ = self._pick_ticker(close)
        r = 100.0 * close[self.ticker_].pct_change(fill_method=None).dropna()
        if len(r) < 250:
            raise ValueError("need at least 250 training returns to fit GARCH")
        ar = self.params["direction"] == "ar1"
        am = arch_model(r, mean="ar" if ar else "constant", lags=1 if ar else 0, vol="garch")
        res = am.fit(disp="off", show_warning=False)
        prm = res.params
        self.garch_params_ = {
            "mu": float(prm["Const"] if ar else prm["mu"]),
            "phi": float(prm[f"{r.name}[1]"]) if ar else 0.0,
            "omega": float(prm["omega"]),
            "alpha": float(prm["alpha[1]"]),
            "beta": float(prm["beta[1]"]),
            "sigma2_0": float(r.var()),
        }
        self.is_fitted = True
        return self

    def forecasts(self, data: pd.DataFrame) -> pd.DataFrame:
        """Forecasts made at each t for t+1: ``mean`` (daily) and ``vol`` (annualised)."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        close = to_wide(data, "close")
        r = 100.0 * close[self.ticker_].pct_change(fill_method=None).to_numpy()
        mean_f, var_f = garch_forecast(r, **self.garch_params_)
        return pd.DataFrame(
            {"mean": mean_f / 100.0, "vol": np.sqrt(var_f * TRADING_DAYS) / 100.0},
            index=close.index,
        )

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = to_wide(data, "close")
        f = self.forecasts(data)
        w = np.minimum(p["target_vol"] / f["vol"], p["max_leverage"])
        if p["direction"] == "ar1":
            w = w.where(f["mean"] > 0, 0.0)
        target = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        target[self.ticker_] = w.to_numpy()
        return every_n_rows(target, p["rebalance_days"])
