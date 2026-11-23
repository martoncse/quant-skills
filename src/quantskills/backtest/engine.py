"""Daily close-to-close portfolio backtester.

Timing convention (no look-ahead):
    * Target weights in row *t* are decided with information up to the close of *t*
      and traded at that close.
    * They earn the asset returns from close *t* to close *t+1*.
    * A row that is entirely NaN means "no rebalance": positions drift with prices.

Between rebalances weights drift, so turnover and costs are measured against the
*drifted* book, not the stale target. Costs are charged in NAV terms:

    cost_t = turnover_t * (commission_bps + slippage_bps) / 1e4
    turnover_t = sum_i |w_target[i,t] - w_drifted[i,t]|
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Linear costs per unit of traded notional (both sides).

    Defaults (5 bps commission + 5 bps slippage) are deliberately non-zero;
    pass ``CostModel(0, 0)`` explicitly to study a frictionless benchmark.
    """

    commission_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    returns: pd.Series  # net daily returns
    gross_returns: pd.Series
    costs: pd.Series  # cost drag in return units
    turnover: pd.Series
    weights: pd.DataFrame  # post-trade weights at each close
    cost_model: CostModel

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    def split(self, date: str | pd.Timestamp) -> tuple[BacktestResult, BacktestResult]:
        """Split into (<= date, > date) pieces, e.g. in-sample vs out-of-sample."""
        ts = pd.Timestamp(date)
        idx = self.returns.index
        return self._subset(idx <= ts), self._subset(idx > ts)

    def trim_warmup(self) -> BacktestResult:
        """Drop leading days with no positions (e.g. while a lookback fills)."""
        exposed = (self.weights.abs().sum(axis=1) > 0).to_numpy()
        if not exposed.any():
            return self
        mask = np.zeros(len(exposed), dtype=bool)
        mask[int(np.argmax(exposed)) :] = True
        return self._subset(mask)

    def _subset(self, mask: np.ndarray) -> BacktestResult:
        return BacktestResult(
            returns=self.returns[mask],
            gross_returns=self.gross_returns[mask],
            costs=self.costs[mask],
            turnover=self.turnover[mask],
            weights=self.weights[mask],
            cost_model=self.cost_model,
        )


def run_backtest(
    target_weights: pd.DataFrame,
    close: pd.DataFrame,
    costs: CostModel | None = None,
) -> BacktestResult:
    """Simulate holding ``target_weights`` (date x ticker, fractions of NAV)."""
    costs = costs if costs is not None else CostModel()
    close = close.sort_index()
    targets = target_weights.reindex(index=close.index, columns=close.columns)
    asset_ret = close.pct_change(fill_method=None).to_numpy(dtype=float)
    tgt = targets.to_numpy(dtype=float)
    rebalance = ~np.all(np.isnan(tgt), axis=1)

    n_days, n_assets = tgt.shape
    held = np.zeros(n_assets)  # weights held over (t-1, t]
    post = np.zeros((n_days, n_assets))
    gross = np.zeros(n_days)
    turnover = np.zeros(n_days)

    for t in range(n_days):
        r = np.nan_to_num(asset_ret[t], nan=0.0) if t > 0 else np.zeros(n_assets)
        g = float(held @ r)
        gross[t] = g
        drifted = held * (1.0 + r) / (1.0 + g) if (1.0 + g) > 0 else np.zeros(n_assets)
        new = np.nan_to_num(tgt[t], nan=0.0) if rebalance[t] else drifted
        turnover[t] = float(np.abs(new - drifted).sum())
        post[t] = new
        held = new

    idx = close.index
    cost = turnover * costs.rate
    return BacktestResult(
        returns=pd.Series(gross - cost, index=idx, name="net"),
        gross_returns=pd.Series(gross, index=idx, name="gross"),
        costs=pd.Series(cost, index=idx, name="costs"),
        turnover=pd.Series(turnover, index=idx, name="turnover"),
        weights=pd.DataFrame(post, index=idx, columns=close.columns),
        cost_model=costs,
    )
