"""Strategy #1: cross-sectional factor long-short.

Hypothesis: a blend of price factors (momentum, short-term reversal, low
volatility) ranks next-month returns better than chance.

Method:
    1. Compute each factor and z-score it across tickers each day.
    2. ``fit`` measures each factor's mean Spearman IC against ``horizon``-day
       forward returns *on the training window only* and uses the sign and size
       of those ICs as blend weights (``weighting="ic"``), or weights equally.
    3. Every ``rebalance_days``, go long the top ``quantile`` of the blended
       score and short the bottom ``quantile``, equal-weighted, dollar neutral,
       with ``gross`` total exposure.

Book reference: ML4T 3rd ed., ch. 7 (labels, IC) and ch. 8 (price features).
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.eval.ic import information_coefficient
from quantskills.features.factors import compute_factors, forward_returns, zscore_cross_section
from quantskills.strategies.base import Strategy


class FactorLongShort(Strategy):
    """Long top / short bottom quantile of an IC-weighted blend of price factors."""

    name = "factor-long-short"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "factors": ("momentum", "reversal", "low_vol"),
            "weighting": "ic",  # "ic" or "equal"
            "horizon": 21,
            "rebalance_days": 21,
            "quantile": 0.2,
            "gross": 1.0,
            "min_assets": 10,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["weighting"] not in {"ic", "equal"}:
            raise ValueError("weighting must be 'ic' or 'equal'")
        if not 0 < self.params["quantile"] <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        self.factor_weights_: dict[str, float] = {}
        self.train_ic_: dict[str, float] = {}

    def _scores(self, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
        raw = compute_factors(close, self.params["factors"])
        return {k: zscore_cross_section(v) for k, v in raw.items()}

    def fit(self, data: pd.DataFrame) -> Self:
        close = to_wide(data, "close")
        scores = self._scores(close)
        h = self.params["horizon"]
        # Labels come from the training window only, so the last h rows are NaN
        # and no information leaks past the end of ``data``.
        fwd = forward_returns(close, h)
        ics = {}
        for name, z in scores.items():
            ic = information_coefficient(z, fwd)
            ics[name] = float(ic.iloc[::h].mean()) if len(ic) else 0.0  # non-overlapping
        self.train_ic_ = ics
        if self.params["weighting"] == "equal":
            weights = {k: 1.0 / len(ics) for k in ics}
        else:
            total = sum(abs(v) for v in ics.values())
            weights = (
                {k: v / total for k, v in ics.items()}
                if total > 0
                else {k: 1.0 / len(ics) for k in ics}
            )
        self.factor_weights_ = weights
        self.is_fitted = True
        return self

    def composite(self, data: pd.DataFrame) -> pd.DataFrame:
        """Blended factor score, wide ``date x ticker``."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        close = to_wide(data, "close")
        scores = self._scores(close)
        parts = [scores[k] * w for k, w in self.factor_weights_.items()]
        stacked = np.stack([p.to_numpy() for p in parts])
        # Require every factor to be present for a ticker to be scored.
        blended = np.where(np.isnan(stacked).any(axis=0), np.nan, stacked.sum(axis=0))
        return pd.DataFrame(blended, index=close.index, columns=close.columns)

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        score = self.composite(data)
        q, gross = self.params["quantile"], self.params["gross"]
        valid = score.notna().sum(axis=1)
        tradable = valid >= self.params["min_assets"]
        weights = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
        if not tradable.any():
            return weights.fillna(0.0)

        first = int(np.argmax(tradable.to_numpy()))
        step = self.params["rebalance_days"]
        rebal_dates = score.index[first::step]

        ranks = score.loc[rebal_dates].rank(axis=1, pct=True)
        longs = ranks > 1.0 - q
        shorts = ranks <= q
        n_long = longs.sum(axis=1).replace(0, np.nan)
        n_short = shorts.sum(axis=1).replace(0, np.nan)
        w = longs.astype(float).div(n_long, axis=0) - shorts.astype(float).div(n_short, axis=0)
        w = (w * gross / 2.0).fillna(0.0)
        w.loc[~tradable.loc[rebal_dates]] = 0.0
        weights.loc[rebal_dates] = w
        weights.iloc[:first] = 0.0
        return weights
