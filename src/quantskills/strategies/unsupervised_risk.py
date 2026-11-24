"""Strategy #7: unsupervised risk factors (PCA eigenportfolios, HRP clustering).

Hypothesis: allocating by the *structure* of risk, learned without labels from
past returns, gives a steadier portfolio than naive weights, without needing
return forecasts (the noisiest input in portfolio construction).

Method:
    Every ``rebalance_days``, using only the trailing ``lookback`` days of returns:

    * ``hrp``: hierarchical clustering of the correlation matrix, then recursive
      bisection by inverse variance (López de Prado, 2016).
    * ``eigen``: the k-th PCA eigenportfolio (``eigen_k=0`` is the market-like
      first component; higher k are long-short statistical factors).
    * ``min_var`` / ``inverse_vol`` / ``equal``: baselines to compare against.

    Weights are scaled to ``gross``. Nothing is learned in ``fit``: the model
    re-estimates on a rolling window, so it is out-of-sample by construction.

Book reference: ML4T 3rd ed., ch. 14 (PCA, latent factors) and ch. 17 (HRP,
portfolio construction).
"""

from __future__ import annotations

from typing import Any, Self

import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.features.factors import simple_returns
from quantskills.portfolio import ALLOCATORS, allocate
from quantskills.strategies.base import Strategy


class UnsupervisedRisk(Strategy):
    """Rolling HRP / PCA-eigenportfolio allocation learned from past returns only."""

    name = "unsupervised-risk"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "method": "hrp",
            "lookback": 252,
            "rebalance_days": 21,
            "eigen_k": 0,
            "shrink": True,
            "gross": 1.0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["method"] not in ALLOCATORS:
            raise ValueError(f"method must be one of {ALLOCATORS}")
        if self.params["lookback"] < 20:
            raise ValueError("lookback must be >= 20 days")

    def fit(self, data: pd.DataFrame) -> Self:
        self.is_fitted = True  # rolling estimation; nothing to learn up front
        return self

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        p = self.params
        rets = simple_returns(to_wide(data, "close"))
        out = pd.DataFrame(float("nan"), index=rets.index, columns=rets.columns)
        lb = p["lookback"]
        out.iloc[:lb] = 0.0
        for i in range(lb, len(rets), p["rebalance_days"]):
            window = rets.iloc[i - lb + 1 : i + 1]  # rows up to and including t
            try:
                w = allocate(window, p["method"], shrink=p["shrink"], eigen_k=p["eigen_k"])
            except ValueError:
                out.iloc[i] = 0.0
                continue
            out.iloc[i] = (w * p["gross"]).to_numpy()
        return out
