"""One-call risk snapshot of a price panel, used by the terminal's RISK screen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.eval.metrics import TRADING_DAYS
from quantskills.features.factors import simple_returns
from quantskills.portfolio.allocate import allocate
from quantskills.portfolio.covariance import cov_to_corr, ledoit_wolf_cov
from quantskills.portfolio.hrp import cluster_order
from quantskills.portfolio.risk import risk_contributions, risk_report


@dataclass
class RiskView:
    method: str
    lookback: int
    start: pd.Timestamp
    end: pd.Timestamp
    table: pd.DataFrame  # ticker -> weight, ann_vol, risk_contrib (sorted by weight)
    metrics: dict[str, float]
    corr: pd.DataFrame  # top holdings, in cluster order


def risk_view(
    prices: pd.DataFrame, method: str = "hrp", lookback: int = 252, max_corr: int = 12
) -> RiskView:
    """Weights from ``method`` on the last ``lookback`` days, their VaR/CVaR and correlations.

    Historical, in-sample risk of today's weights: a description of the recent
    past, not a forecast.
    """
    if lookback < 20:
        raise ValueError("lookback must be >= 20 days")
    rets = simple_returns(to_wide(prices, "close")).iloc[1:].iloc[-lookback:]
    rets = rets.dropna(axis=1, how="any")
    if rets.shape[1] < 2:
        raise ValueError("need at least 2 tickers with a full return window")
    w = allocate(rets, method)
    cov = ledoit_wolf_cov(rets)
    table = pd.DataFrame(
        {
            "weight": w,
            "ann_vol": rets.std() * np.sqrt(TRADING_DAYS),
            "risk_contrib": risk_contributions(w, cov),
        }
    ).sort_values("weight", key=np.abs, ascending=False)
    top = list(table.index[:max_corr])
    corr = cov_to_corr(cov).loc[top, top]
    order = cluster_order(corr)
    return RiskView(
        method=method,
        lookback=len(rets),
        start=pd.Timestamp(rets.index[0]),
        end=pd.Timestamp(rets.index[-1]),
        table=table,
        metrics=risk_report(rets, w),
        corr=corr.loc[order, order],
    )
