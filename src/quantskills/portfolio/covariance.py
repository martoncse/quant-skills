"""Covariance estimators. Inputs are wide ``date x ticker`` simple returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from quantskills.eval.metrics import TRADING_DAYS


def _clean(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.dropna(how="any")
    if len(r) < 2:
        raise ValueError("need at least 2 complete rows of returns")
    return r


def sample_cov(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Sample covariance of complete rows, optionally x252."""
    r = _clean(returns)
    cov = r.cov()
    return cov * TRADING_DAYS if annualize else cov


def ledoit_wolf_cov(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage towards a scaled identity (Ledoit & Wolf, 2004).

    With few observations per asset the sample covariance is noisy and
    optimisers chase that noise; shrinkage pulls extreme eigenvalues in.
    """
    r = _clean(returns)
    est = LedoitWolf().fit(r.to_numpy())
    cov = pd.DataFrame(est.covariance_, index=r.columns, columns=r.columns)
    return cov * TRADING_DAYS if annualize else cov


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(np.clip(corr, -1.0, 1.0), index=cov.index, columns=cov.columns)
