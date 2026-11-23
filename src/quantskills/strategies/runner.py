"""Fit-on-train, evaluate-on-holdout runner shared by all strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantskills.backtest.engine import BacktestResult, CostModel, run_backtest
from quantskills.data.loader import to_wide, validate_prices
from quantskills.eval.metrics import sharpe_ratio
from quantskills.strategies.base import Strategy


@dataclass
class StrategyRun:
    strategy: Strategy
    train_end: pd.Timestamp
    full: BacktestResult
    in_sample: BacktestResult
    out_of_sample: BacktestResult


def run_strategy(
    strategy: Strategy,
    data: pd.DataFrame,
    train_end: str | pd.Timestamp,
    costs: CostModel | None = None,
) -> StrategyRun:
    """Fit on dates <= ``train_end``, then backtest over the full period.

    Signals for the hold-out use features computed from past prices only, and
    model parameters frozen at ``train_end``. Look at ``out_of_sample`` once per
    experiment: re-tuning after peeking turns it into in-sample data.
    """
    validate_prices(data)
    ts = pd.Timestamp(train_end)
    dates = data.index.get_level_values("date")
    if not (dates.min() < ts < dates.max()):
        raise ValueError("train_end must fall strictly inside the data's date range")
    strategy.fit(data[dates <= ts])
    weights = strategy.signals(data)
    result = run_backtest(weights, to_wide(data, "close"), costs)
    is_, oos = result.split(ts)
    return StrategyRun(strategy, ts, result, is_, oos)


def sweep_params(
    strategy_cls: type[Strategy],
    grid: list[dict[str, Any]],
    data: pd.DataFrame,
    train_end: str | pd.Timestamp,
    costs: CostModel | None = None,
) -> pd.DataFrame:
    """Score every parameter set on the **training window only**.

    Returns one row per trial with its in-sample Sharpe. Pass
    ``table["is_sharpe"]`` as ``trial_sharpes`` to the deflated Sharpe ratio so
    the selection is penalised for every configuration tried.
    """
    ts = pd.Timestamp(train_end)
    dates = data.index.get_level_values("date")
    train = data[dates <= ts]
    close = to_wide(train, "close")
    rows = []
    for params in grid:
        strat = strategy_cls(**params).fit(train)
        res = run_backtest(strat.signals(train), close, costs)
        rows.append({**params, "is_sharpe": sharpe_ratio(res.trim_warmup().returns)})
    return pd.DataFrame(rows)
