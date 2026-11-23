"""Synthetic price panels for offline tests and demos.

Returns follow a one-factor model with a slowly varying, persistent per-asset
drift (AR(1)), which plants a weak, *known* momentum effect:

    r[i,t] = beta_i * m_t + mu[i,t] + e[i,t],   mu[i,t] = phi * mu[i,t-1] + u[i,t]

Results on this data are illustrations of the machinery, not evidence about markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantskills.data.loader import PRICE_COLUMNS


def synthetic_prices(
    n_assets: int = 30,
    n_days: int = 1260,
    seed: int = 0,
    start: str = "2015-01-02",
    drift_vol: float = 0.0004,
    drift_persistence: float = 0.995,
    idio_vol: float = 0.015,
) -> pd.DataFrame:
    """Generate a long-format OHLCV panel (``MultiIndex(date, ticker)``).

    ``drift_vol`` is the stationary daily std of the persistent drift; set it
    to 0 for a panel with no planted signal (a pure-noise null).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    tickers = [f"SYN{i:02d}" for i in range(n_assets)]

    market = rng.normal(0.0003, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_assets)
    innov_sd = drift_vol * np.sqrt(1.0 - drift_persistence**2)
    drift = np.empty((n_days, n_assets))
    drift[0] = rng.normal(0.0, drift_vol, n_assets)
    shocks = rng.normal(0.0, innov_sd, (n_days, n_assets))
    for t in range(1, n_days):
        drift[t] = drift_persistence * drift[t - 1] + shocks[t]
    rets = market[:, None] * beta[None, :] + drift + rng.normal(0, idio_vol, (n_days, n_assets))
    rets = np.clip(rets, -0.5, 0.5)

    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    prev = np.vstack([np.full(n_assets, 100.0), close[:-1]])
    open_ = prev * (1.0 + rng.normal(0, 0.002, (n_days, n_assets)))
    spread = np.abs(rng.normal(0, 0.005, (n_days, n_assets)))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(13.0, 0.4, (n_days, n_assets)).round()

    fields = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = {k: v.reshape(-1) for k, v in fields.items()}
    return pd.DataFrame(data, index=idx)[PRICE_COLUMNS]
