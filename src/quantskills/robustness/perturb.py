"""Perturbation tests: does a strategy survive small changes it should not care about?

Three probes, all run on the **training window** so they never touch the hold-out.
By default each one is scored **walk-forward** inside that window (fit on the past,
trade the next block), because scoring a model on the days it was fitted to
flatters anything flexible enough to memorise them:

* **Price noise.** Multiply every OHLC price by ``exp(eps)``, ``eps ~ N(0, bps/1e4)``,
  refit and generate signals on the noisy prices, then trade at the *clean* prices.
  A real edge should survive data errors of a few basis points. (P&L is measured on
  clean prices on purpose: on noisy prices a reversal rule would "profit" from the
  noise itself.)
* **Execution delay.** Trade the same target weights ``k`` days late. An edge that
  disappears after one day of delay was probably an artefact of perfect timing.
* **Parameter sensitivity.** Sharpe over a small grid of two parameters. A good
  configuration sits on a plateau, not on a lone spike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from quantskills.backtest.engine import BacktestResult, CostModel, run_backtest
from quantskills.data.loader import to_wide
from quantskills.eval.metrics import sharpe_ratio
from quantskills.strategies.base import Strategy
from quantskills.utils.splits import walk_forward_splits

PRICE_FIELDS = ("open", "high", "low", "close", "adj_close")


def add_price_noise(prices: pd.DataFrame, bps: float, seed: int = 0) -> pd.DataFrame:
    """Multiplicative log-normal noise on every price field, one draw per (date, ticker).

    The same factor scales open/high/low/close in a row, so ``low <= close <= high``
    still holds. Volume is untouched.
    """
    if bps < 0:
        raise ValueError("bps must be >= 0")
    out = prices.copy()
    if bps == 0:
        return out
    rng = np.random.default_rng(seed)
    factor = np.exp(rng.normal(0.0, bps / 1e4, size=len(out)))
    for col in PRICE_FIELDS:
        if col in out.columns:
            out[col] = out[col].to_numpy(dtype=float) * factor
    return out


def delay_weights(weights: pd.DataFrame, days: int) -> pd.DataFrame:
    """Shift target weights ``days`` rows later (trade late). Leading rows become NaN = hold."""
    if days < 0:
        raise ValueError("days must be >= 0 (a negative delay would be look-ahead)")
    return weights.shift(days)


EVAL_MODES = ("walk-forward", "in-sample")


def insample_backtest(
    strategy: Strategy,
    train: pd.DataFrame,
    costs: CostModel | None = None,
    trade_close: pd.DataFrame | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Fit on ``train``, signal on ``train``, backtest at ``trade_close`` (default: train close).

    Returns the warm-up-trimmed result and the raw target weights. The model has
    seen every day it is scored on, so this flatters anything that can memorise.
    """
    strategy.fit(train)
    weights = strategy.signals(train)
    close = trade_close if trade_close is not None else to_wide(train, "close")
    return run_backtest(weights, close, costs).trim_warmup(), weights


def walk_forward_backtest(
    strategy_cls: type[Strategy],
    params: Mapping[str, Any],
    train: pd.DataFrame,
    costs: CostModel | None = None,
    n_folds: int = 4,
    min_train_frac: float = 0.5,
    trade_close: pd.DataFrame | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Expanding-window walk-forward *inside* the training window.

    For each fold: fit on every date before the fold, generate signals with data up
    to the fold's end, keep the fold's rows. The stitched weights are backtested
    once, so turnover at fold boundaries is charged. Every scored day is out-of-fold
    for the model that traded it: a pseudo-out-of-sample that leaves the hold-out alone.
    """
    dates = train.index.get_level_values("date")
    close = trade_close if trade_close is not None else to_wide(train, "close")
    min_train = int(len(close) * min_train_frac)
    stitched = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    first_test = None
    for fit_dates, test_dates in walk_forward_splits(close.index, n_folds, 1, min_train):
        strategy = strategy_cls(**params).fit(train[dates <= fit_dates[-1]])
        w = strategy.signals(train[dates <= test_dates[-1]]).reindex(columns=close.columns)
        block = w.loc[test_dates].copy()
        # Start each fold from the new model's current target, not the old model's book.
        block.iloc[0] = w.ffill().loc[test_dates[0]].fillna(0.0)
        stitched.loc[test_dates] = block.to_numpy()
        first_test = test_dates[0] if first_test is None else first_test
    if first_test is None:
        raise ValueError("not enough training dates for walk-forward folds")
    stitched.loc[: first_test - pd.Timedelta(days=1)] = 0.0
    res = run_backtest(stitched, close, costs)
    before = res.returns.index[res.returns.index < first_test]
    oof = res.split(before[-1])[1] if len(before) else res
    return oof.trim_warmup(), stitched


def train_backtest(
    strategy_cls: type[Strategy],
    params: Mapping[str, Any],
    train: pd.DataFrame,
    costs: CostModel | None = None,
    mode: str = "walk-forward",
    n_folds: int = 4,
    trade_close: pd.DataFrame | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Backtest on the training window, ``mode`` = ``"walk-forward"`` or ``"in-sample"``."""
    if mode == "walk-forward":
        return walk_forward_backtest(
            strategy_cls, params, train, costs, n_folds, trade_close=trade_close
        )
    if mode == "in-sample":
        return insample_backtest(strategy_cls(**params), train, costs, trade_close)
    raise ValueError(f"mode must be one of {EVAL_MODES}")


def noise_test(
    strategy_cls: type[Strategy],
    params: Mapping[str, Any],
    train: pd.DataFrame,
    levels_bps: Sequence[float] = (10.0, 25.0, 50.0),
    seeds: Sequence[int] = (0, 1),
    costs: CostModel | None = None,
    mode: str = "walk-forward",
    n_folds: int = 4,
) -> pd.DataFrame:
    """Sharpe after refitting on noisy prices, traded at clean prices. One row per (bps, seed)."""
    clean_close = to_wide(train, "close")
    rows = []
    for bps, seed in product(levels_bps, seeds):
        noisy = add_price_noise(train, bps, seed)
        try:
            res, _ = train_backtest(
                strategy_cls, params, noisy, costs, mode, n_folds, trade_close=clean_close
            )
            sr = sharpe_ratio(res.returns)
        except ValueError:  # e.g. a pairs strategy finds no pair in the noisy data
            sr = float("nan")
        rows.append({"bps": float(bps), "seed": seed, "sharpe": sr})
    return pd.DataFrame(rows)


def delay_test(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    delays: Sequence[int] = (0, 1, 2, 5),
    costs: CostModel | None = None,
) -> pd.DataFrame:
    """Sharpe of the same target weights traded ``k`` days late."""
    rows = []
    for d in delays:
        res = run_backtest(delay_weights(weights, d), close, costs).trim_warmup()
        rows.append({"delay": int(d), "sharpe": sharpe_ratio(res.returns)})
    return pd.DataFrame(rows)


def sensitivity_surface(
    strategy_cls: type[Strategy],
    base_params: Mapping[str, Any],
    axes: Mapping[str, Sequence[Any]],
    train: pd.DataFrame,
    costs: CostModel | None = None,
    mode: str = "walk-forward",
    n_folds: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sharpe over the grid ``axes`` (exactly two parameters), training window only.

    Returns ``(surface, trial_returns)``: ``surface`` is a 2-D table of Sharpe
    (rows = first parameter, columns = second); ``trial_returns`` is the ``T x N``
    daily-return matrix of every grid point, ready for :func:`pbo_cscv` and for the
    deflated Sharpe's ``trial_sharpes``.
    """
    if len(axes) != 2:
        raise ValueError("axes must name exactly two parameters")
    (p1, v1), (p2, v2) = axes.items()
    surface = pd.DataFrame(np.nan, index=pd.Index(list(v1), name=p1), columns=list(v2))
    surface.columns.name = p2
    series: dict[str, pd.Series] = {}
    for a, b in product(v1, v2):
        params = {**base_params, p1: a, p2: b}
        try:
            strategy_cls(**params)
        except ValueError:  # an invalid combination (e.g. exit_z >= entry_z)
            continue
        res, _ = train_backtest(strategy_cls, params, train, costs, mode, n_folds)
        series[f"{p1}={a!r},{p2}={b!r}"] = res.returns
        surface.loc[a, b] = sharpe_ratio(res.returns)
    # Common dates only, so every config is compared over the same days.
    trials = pd.DataFrame(series).dropna()
    return surface, trials
