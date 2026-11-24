"""PCA risk factors and eigenportfolios (ML4T ch. 14; Avellaneda & Lee, 2010).

PCA on the correlation matrix of returns finds uncorrelated statistical factors.
The first usually looks like "the market" (all loadings the same sign); later ones
are long-short spreads between groups of assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    eigenvalues: pd.Series  # sorted, largest first
    explained: pd.Series  # fraction of total variance
    loadings: pd.DataFrame  # ticker x component (unit eigenvectors)
    vol: pd.Series  # per-asset return std used for standardising


def pca_factors(returns: pd.DataFrame, n_components: int | None = None) -> PCAResult:
    """Eigen-decompose the correlation matrix of complete rows of ``returns``."""
    r = returns.dropna(how="any")
    if len(r) < 3:
        raise ValueError("need at least 3 complete rows of returns")
    vol = r.std()
    corr = np.corrcoef(r.to_numpy(), rowvar=False)
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Fix signs so each eigenvector's loadings sum to a positive number (deterministic output).
    signs = np.where(vecs.sum(axis=0) < 0, -1.0, 1.0)
    vecs = vecs * signs
    k = len(vals) if n_components is None else min(n_components, len(vals))
    names = [f"PC{i + 1}" for i in range(k)]
    return PCAResult(
        eigenvalues=pd.Series(vals[:k], index=names),
        explained=pd.Series(vals[:k] / vals.sum(), index=names),
        loadings=pd.DataFrame(vecs[:, :k], index=r.columns, columns=names),
        vol=vol,
    )


def eigenportfolio(returns: pd.DataFrame, k: int = 0, gross: float = 1.0) -> pd.Series:
    """Weights of the k-th eigenportfolio: w_i = v_ik / sigma_i, scaled to ``gross``.

    Dividing by sigma converts correlation-space loadings into dollar weights.
    """
    res = pca_factors(returns)
    if not 0 <= k < res.loadings.shape[1]:
        raise ValueError(f"k must be in [0, {res.loadings.shape[1] - 1}]")
    raw = res.loadings.iloc[:, k] / res.vol.replace(0.0, np.nan)
    raw = raw.fillna(0.0)
    total = raw.abs().sum()
    return raw * (gross / total) if total > 0 else raw
