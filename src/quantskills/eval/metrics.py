"""Return-series performance metrics (simple daily returns in, annualised out)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric CAGR: prod(1+r)^(periods/n) - 1."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(r)) - 1.0


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe: mean(r - rf/periods) / std(r) * sqrt(periods)."""
    r = returns.dropna() - rf / periods
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0 or np.isnan(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """mean(r) / downside deviation * sqrt(periods), target 0."""
    r = returns.dropna()
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    if downside == 0:
        return float("nan")
    return float(r.mean() / downside * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Equity / running peak - 1 (<= 0)."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Most negative drawdown, e.g. -0.25 for a 25% peak-to-trough loss."""
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    return float("nan") if mdd == 0 else annualized_return(returns, periods) / abs(mdd)


def summary(returns: pd.Series, periods: int = TRADING_DAYS) -> dict[str, float]:
    r = returns.dropna()
    return {
        "ann_return": annualized_return(r, periods),
        "ann_vol": annualized_volatility(r, periods),
        "sharpe": sharpe_ratio(r, periods=periods),
        "sortino": sortino_ratio(r, periods),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, periods),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "n_days": float(len(r)),
    }
