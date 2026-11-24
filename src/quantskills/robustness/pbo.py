"""Probability of backtest overfitting via combinatorially symmetric cross-validation.

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of
Backtest Overfitting", *Journal of Computational Finance*.

Idea in plain words: you tried N configurations. Cut the history into S blocks.
For every way of picking half the blocks as "in-sample" (IS), choose the config
with the best IS Sharpe and see where it ranks on the other half (OOS). If the
IS winner usually lands in the bottom half OOS, the selection process is fitting
noise. PBO is the fraction of splits where that happens.

    omega  = rank of the IS winner among the N configs OOS / (N + 1)
    lambda = log(omega / (1 - omega))          (logit; > 0 means above the OOS median)
    PBO    = P(lambda <= 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from quantskills.eval.metrics import TRADING_DAYS


@dataclass(frozen=True)
class PBOResult:
    pbo: float  # P(IS winner ranks at or below the OOS median)
    logits: np.ndarray  # lambda per split
    is_sharpe: np.ndarray  # annualised IS Sharpe of the IS winner, per split
    oos_sharpe: np.ndarray  # annualised OOS Sharpe of that same config, per split
    prob_oos_loss: float  # P(IS winner has OOS Sharpe < 0)
    degradation_slope: float  # OLS slope of OOS on IS Sharpe (negative = overfit)
    n_trials: int
    n_blocks: int
    n_splits: int


def _block_stats(m: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = np.array_split(m, n_blocks, axis=0)
    s1 = np.stack([b.sum(axis=0) for b in blocks])  # (S, N)
    s2 = np.stack([(b**2).sum(axis=0) for b in blocks])
    n = np.array([len(b) for b in blocks], dtype=float)
    return s1, s2, n


def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Per-period Sharpe from sums (rows = splits, cols = trials)."""
    n = n[:, None]
    mean = s1 / n
    var = (s2 - n * mean**2) / np.maximum(n - 1, 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mean / sd, 0.0)
    return sr


def pbo_cscv(
    trial_returns: pd.DataFrame | np.ndarray,
    n_blocks: int = 16,
    periods: int = TRADING_DAYS,
) -> PBOResult:
    """PBO for the selection "pick the config with the best Sharpe".

    ``trial_returns`` is ``T x N``: one column of daily returns per configuration,
    all over the same dates (the **training window**: never feed it the hold-out).
    ``n_blocks`` must be even; S = 16 gives C(16, 8) = 12,870 splits.
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("need a T x N matrix with N >= 2 configurations")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2")
    m = m[~np.isnan(m).all(axis=1)]
    m = np.nan_to_num(m, nan=0.0)
    t, n_trials = m.shape
    if t < 4 * n_blocks:
        raise ValueError(f"only {t} rows for {n_blocks} blocks; need >= {4 * n_blocks}")

    s1, s2, cnt = _block_stats(m, n_blocks)
    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    mask = np.zeros((len(combos), n_blocks))
    np.put_along_axis(mask, combos, 1.0, axis=1)
    inv = 1.0 - mask

    sr_is = _sharpe_from_sums(mask @ s1, mask @ s2, mask @ cnt)
    sr_oos = _sharpe_from_sums(inv @ s1, inv @ s2, inv @ cnt)

    best = np.argmax(sr_is, axis=1)
    rows = np.arange(len(combos))
    oos_ranks = stats.rankdata(sr_oos, axis=1)  # 1 = worst, N = best; ties averaged
    omega = oos_ranks[rows, best] / (n_trials + 1.0)
    logits = np.log(omega / (1.0 - omega))

    ann = np.sqrt(periods)
    is_best = sr_is[rows, best] * ann
    oos_best = sr_oos[rows, best] * ann
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if np.ptp(is_best) > 0 else float("nan")
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        is_sharpe=is_best,
        oos_sharpe=oos_best,
        prob_oos_loss=float(np.mean(oos_best < 0)),
        degradation_slope=slope,
        n_trials=n_trials,
        n_blocks=n_blocks,
        n_splits=len(combos),
    )
