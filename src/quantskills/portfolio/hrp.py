"""Hierarchical Risk Parity (López de Prado, 2016), written from the paper's description.

Three steps:
    1. **Cluster** assets by correlation distance d_ij = sqrt((1 - rho_ij) / 2).
    2. **Quasi-diagonalise**: order assets so that similar ones sit next to each other
       (the leaf order of the dendrogram).
    3. **Recursive bisection**: split the ordered list in halves; give each half
       budget in inverse proportion to its variance (computed with inverse-variance
       weights inside the half), and recurse.

No matrix inversion is needed, so HRP stays stable when the covariance is
nearly singular, which is where min-variance tends to put all its weight in a few names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from quantskills.portfolio.covariance import cov_to_corr


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, None))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def cluster_order(corr: pd.DataFrame, method: str = "single") -> list[str]:
    """Tickers in dendrogram leaf order (steps 1 and 2)."""
    if len(corr) == 1:
        return list(corr.index)
    dist = correlation_distance(corr).to_numpy()
    link = linkage(squareform(dist, checks=False), method=method)
    return [corr.index[i] for i in leaves_list(link)]


def _cluster_variance(cov: np.ndarray) -> float:
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return float(ivp @ cov @ ivp)


def hrp_weights(cov: pd.DataFrame, method: str = "single") -> pd.Series:
    """HRP weights, long-only, summing to 1, indexed like ``cov``."""
    order = cluster_order(cov_to_corr(cov), method)
    c = cov.loc[order, order].to_numpy()
    w = np.ones(len(order))
    stack = [np.arange(len(order))]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        v_left = _cluster_variance(c[np.ix_(left, left)])
        v_right = _cluster_variance(c[np.ix_(right, right)])
        alpha = 1.0 - v_left / (v_left + v_right)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack += [left, right]
    return pd.Series(w, index=order).reindex(cov.index)
