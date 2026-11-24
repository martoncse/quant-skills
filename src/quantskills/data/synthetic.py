"""Synthetic price panels for offline tests and demos.

Returns follow a one-factor model with a slowly varying, persistent per-asset
drift (AR(1)), which plants a weak, *known* momentum effect:

    r[i,t] = beta_i * m_t + mu[i,t] + e[i,t],   mu[i,t] = phi * mu[i,t-1] + u[i,t]

With ``n_pairs > 0``, cointegrated pairs are appended (tickers ``PAIRkA`` /
``PAIRkB``): ``log B = c + h * log A + s`` with a mean-reverting spread
``s_t = kappa * s_{t-1} + eta_t``. They give the pairs strategies something to find.

With ``garch_index=True``, an index ``IDX`` with GARCH(1,1) volatility clustering
is appended (the main panel has constant volatility), for volatility-timing demos.

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
    n_pairs: int = 0,
    pair_kappa: float = 0.93,
    pair_spread_vol: float = 0.008,
    garch_index: bool = False,
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
    fields = _ohlcv(close, rng)

    if n_pairs > 0:
        # A separate stream, so adding pairs never changes the main panel.
        prng = np.random.default_rng([seed, 1])
        pair_close = _pair_closes(market, n_pairs, pair_kappa, pair_spread_vol, prng)
        pair_fields = _ohlcv(pair_close, prng)
        fields = {k: np.hstack([fields[k], pair_fields[k]]) for k in fields}
        tickers += [f"PAIR{k}{side}" for k in range(n_pairs) for side in "AB"]

    if garch_index:
        grng = np.random.default_rng([seed, 2])
        idx_close = 100.0 * np.cumprod(1.0 + _garch_returns(n_days, grng))[:, None]
        idx_fields = _ohlcv(idx_close, grng)
        fields = {k: np.hstack([fields[k], idx_fields[k]]) for k in fields}
        tickers.append("IDX")

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = {k: v.reshape(-1) for k, v in fields.items()}
    return pd.DataFrame(data, index=idx)[PRICE_COLUMNS].sort_index()


def _ohlcv(close: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n_days, n = close.shape
    prev = np.vstack([np.full(n, 100.0), close[:-1]])
    open_ = prev * (1.0 + rng.normal(0, 0.002, (n_days, n)))
    spread = np.abs(rng.normal(0, 0.005, (n_days, n)))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(13.0, 0.4, (n_days, n)).round()
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _garch_returns(
    n_days: int,
    rng: np.random.Generator,
    mu: float = 0.0004,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
) -> np.ndarray:
    """r_t = mu + sigma_t z_t,  sigma^2_t = omega + alpha (r_{t-1}-mu)^2 + beta sigma^2_{t-1}."""
    z = rng.standard_normal(n_days)
    r = np.empty(n_days)
    s2 = omega / (1.0 - alpha - beta)
    eps = 0.0
    for t in range(n_days):
        s2 = omega + alpha * eps**2 + beta * s2
        eps = np.sqrt(s2) * z[t]
        r[t] = mu + eps
    return np.clip(r, -0.5, 0.5)


def _pair_closes(
    market: np.ndarray, n_pairs: int, kappa: float, spread_vol: float, rng: np.random.Generator
) -> np.ndarray:
    """Closes for ``n_pairs`` cointegrated pairs, columns ordered A0, B0, A1, B1, ..."""
    n_days = len(market)
    out = np.empty((n_days, 2 * n_pairs))
    for k in range(n_pairs):
        log_a = np.log(100.0) + np.cumsum(
            market * rng.uniform(0.7, 1.3) + rng.normal(0, 0.012, n_days)
        )
        hedge = rng.uniform(0.7, 1.3)
        s = np.empty(n_days)
        s[0] = 0.0
        eta = rng.normal(0, spread_vol, n_days)
        for t in range(1, n_days):
            s[t] = kappa * s[t - 1] + eta[t]
        log_b = np.log(100.0) * (1 - hedge) + hedge * log_a + s
        out[:, 2 * k] = np.exp(log_a)
        out[:, 2 * k + 1] = np.exp(log_b)
    return out


def stylized_prices(
    n_assets: int = 5,
    n_days: int = 1260,
    seed: int = 0,
    start: str = "2015-01-02",
    alpha: float = 0.08,
    beta: float = 0.90,
    dof: float = 5.0,
) -> pd.DataFrame:
    """A small panel with the textbook stylized facts, for testing generators.

    A market factor with GARCH(1,1) variance (volatility clustering) and Student-t
    shocks (fat tails); each asset = beta_i x market + Student-t idiosyncratic noise
    scaled by the same volatility state. No planted alpha.
    """
    rng = np.random.default_rng([seed, 3])
    omega = 0.01**2 * (1.0 - alpha - beta)
    t_scale = np.sqrt((dof - 2.0) / dof)
    var = np.empty(n_days)
    m = np.empty(n_days)
    v = omega / (1.0 - alpha - beta)
    eps_prev = 0.0
    for t in range(n_days):
        v = omega + alpha * eps_prev**2 + beta * v
        var[t] = v
        eps_prev = np.sqrt(v) * rng.standard_t(dof) * t_scale
        m[t] = eps_prev
    betas = rng.uniform(0.7, 1.3, n_assets)
    vol_state = np.sqrt(var / var.mean())[:, None]
    idio = 0.008 * vol_state * rng.standard_t(dof, (n_days, n_assets)) * t_scale
    r = 0.0002 + m[:, None] * betas + idio
    close = 100.0 * np.cumprod(1.0 + r, axis=0)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    tickers = [f"STY{i}" for i in range(n_assets)]
    fields = _ohlcv(close, rng)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame({k: v.reshape(-1) for k, v in fields.items()}, index=idx)


def trend_regime_prices(
    n_days: int = 1260,
    seed: int = 0,
    drift: float = 0.003,
    switch: float = 0.01,
    vol: float = 0.01,
    ticker: str = "TRND",
    start: str = "2015-01-02",
) -> tuple[pd.DataFrame, pd.Series]:
    """One asset whose drift flips between +drift and -drift (each day w.p. ``switch``).

    A positive control for trend-following learners (strategy #12): following the
    regime pays, and a 20-day momentum rule catches most of it. Returns the long OHLCV
    panel and the true regime (+1 / -1) per day, for diagnostics only.
    """
    rng = np.random.default_rng([seed, 4])
    state = np.empty(n_days)
    s = 1.0
    for t in range(n_days):
        if rng.uniform() < switch:
            s = -s
        state[t] = s
    close = 100.0 * np.cumprod(1.0 + drift * state + rng.normal(0.0, vol, n_days))
    fields = _ohlcv(close[:, None], rng)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    idx = pd.MultiIndex.from_product([dates, [ticker]], names=["date", "ticker"])
    panel = pd.DataFrame({k: v.reshape(-1) for k, v in fields.items()}, index=idx)
    return panel, pd.Series(state, index=dates, name="regime")
