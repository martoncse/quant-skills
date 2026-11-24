"""Pair selection (cointegration tests) and spread trading rules.

Shared by strategy #3 (fixed hedge ratio) and #5 (Kalman / Bayesian hedge ratio).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass(frozen=True)
class Pair:
    y: str  # the leg we hold +1 unit of when long the spread
    x: str  # the hedge leg
    hedge: float  # log y ~ intercept + hedge * log x
    intercept: float
    stat: float  # test statistic (EG t-stat, or Johansen trace / 95% critical value)
    pvalue: float  # EG p-value; NaN for Johansen


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    (c, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(c), float(b)


def candidate_pairs(log_close: pd.DataFrame, max_candidates: int) -> list[tuple[str, str]]:
    """Most correlated pairs by daily log-return correlation (a cheap pre-filter)."""
    rets = log_close.diff().dropna(how="all")
    corr = rets.corr().to_numpy()
    cols = list(log_close.columns)
    scored = [(corr[i, j], cols[i], cols[j]) for i, j in combinations(range(len(cols)), 2)]
    scored = [s for s in scored if np.isfinite(s[0])]
    scored.sort(reverse=True)
    return [(a, b) for _, a, b in scored[:max_candidates]]


def coint_test_pair(log_close: pd.DataFrame, a: str, b: str, method: str) -> Pair | None:
    """Cointegration test of one pair on the given (training) window."""
    df = log_close[[a, b]].dropna()
    if len(df) < 100:
        return None
    ya, yb = df[a].to_numpy(), df[b].to_numpy()
    if method == "engle-granger":
        _, pval, _ = coint(ya, yb)
        c, h = _ols(ya, yb)
        # stat = -log(p) so that "larger is stronger" for both methods
        return Pair(a, b, h, c, float(-np.log(max(pval, 1e-300))), float(pval)) if h > 0 else None
    if method == "johansen":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            res = coint_johansen(np.column_stack([ya, yb]), det_order=0, k_ar_diff=1)
        if np.iscomplexobj(res.eig) and np.abs(np.imag(res.eig)).max() > 1e-12:
            return None  # numerically degenerate pair
        # trace stat (r=0) / critical value (99% column), > 1 means "reject no-cointegration"
        ratio = float(np.real(res.lr1[0]) / res.cvt[0, 2])
        vec = np.real(res.evec[:, 0])
        if vec[0] == 0:
            return None
        h = float(-vec[1] / vec[0])
        if h <= 0:
            return None
        c = float(np.mean(ya - h * yb))
        return Pair(a, b, h, c, ratio, float("nan"))
    raise ValueError("method must be 'engle-granger' or 'johansen'")


def select_pairs(
    log_close: pd.DataFrame,
    method: str = "engle-granger",
    max_pairs: int = 3,
    pvalue: float = 0.05,
    max_candidates: int = 40,
    bonferroni: bool = True,
) -> list[Pair]:
    """Test the most correlated candidates; keep the strongest, each ticker used once.

    Testing many pairs is a multiple-testing problem: with 40 candidates and
    p < 0.05, about 2 false positives are expected by chance alone, and a fluke
    pair loses money out of sample. ``bonferroni=True`` (default) divides the
    p-value threshold by the number of pairs tested. Johansen has no p-value
    here; it uses the stricter 99% critical value instead.
    """
    candidates = candidate_pairs(log_close, max_candidates)
    threshold = pvalue / max(len(candidates), 1) if bonferroni else pvalue
    found = []
    for a, b in candidates:
        p = coint_test_pair(log_close, a, b, method)
        if p is None:
            continue
        ok = p.pvalue < threshold if method == "engle-granger" else p.stat > 1.0
        if ok:
            found.append(p)
    found.sort(key=lambda p: p.stat, reverse=True)
    chosen: list[Pair] = []
    used: set[str] = set()
    for p in found:
        if p.y in used or p.x in used:
            continue
        chosen.append(p)
        used |= {p.y, p.x}
        if len(chosen) == max_pairs:
            break
    return chosen


def band_positions(z: np.ndarray, entry: float, exit_: float, stop: float) -> np.ndarray:
    """Hysteresis rule on a z-score: +1 long spread, -1 short, 0 flat.

    Enter short when z > entry, long when z < -entry; exit when z crosses back
    inside +/- exit; stop out when |z| > stop (the relationship may have broken).
    Causal: the position at t depends on z up to t only.
    """
    pos = np.zeros(len(z))
    state = 0.0
    for t, v in enumerate(z):
        if not np.isfinite(v):
            state = 0.0
        elif state == 0.0:
            if entry < v < stop:
                state = -1.0
            elif -stop < v < -entry:
                state = 1.0
        elif abs(v) > stop or (state > 0 and v > -exit_) or (state < 0 and v < exit_):
            state = 0.0
        pos[t] = state
    return pos


def spread_weights(
    index: pd.Index,
    columns: pd.Index,
    legs: list[tuple[str, str, np.ndarray, np.ndarray]],
    gross: float,
) -> pd.DataFrame:
    """Dollar weights from per-pair positions and hedge ratios.

    For each ``(y, x, position, hedge)``: long the spread is +1 in y and -hedge in x
    (hedging log prices, so the ratio applies to dollar amounts), scaled so each
    pair uses ``gross / n_pairs`` of gross exposure. Emits a row only when some
    position changes (NaN = hold), which keeps turnover and costs down.
    """
    target = pd.DataFrame(0.0, index=index, columns=columns)
    if not legs:
        return target
    budget = gross / len(legs)
    changed = np.zeros(len(index), dtype=bool)
    changed[0] = True
    for y, x, pos, hedge in legs:
        scale = budget / (1.0 + np.abs(hedge))
        target[y] += pos * scale
        target[x] -= pos * hedge * scale
        changed[1:] |= pos[1:] != pos[:-1]
    return target.where(pd.Series(changed, index=index), np.nan)
