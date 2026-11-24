"""Strategy #3: pairs / statistical arbitrage with cointegration.

Hypothesis: some pairs of assets share a long-run equilibrium (their log prices
are cointegrated), so when the spread between them stretches it tends to snap back.

Method:
    1. ``fit`` (training window only): pre-filter the most correlated pairs, test
       each for cointegration (Engle-Granger or Johansen) with a Bonferroni-corrected
       threshold, keep up to ``max_pairs`` with each ticker used once, and freeze
       the hedge ratio.
    2. Spread s_t = log y_t - c - h * log x_t; z-score it with a rolling window of
       ``z_window`` past days (causal).
    3. Short the spread when z > ``entry_z``, buy it when z < -``entry_z``; exit
       inside +/- ``exit_z``; stop out beyond +/- ``stop_z``.

Failure modes: the relationship breaks (regime change, corporate action); many
tested pairs means some "cointegrated" ones are flukes (multiple testing).

Book reference: ML4T 3rd ed., ch. 9 (stationarity, cointegration). Engle & Granger
(1987); Johansen (1991).
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.strategies._pairs import Pair, band_positions, select_pairs, spread_weights
from quantskills.strategies.base import Strategy

PAIR_DEFAULTS: dict[str, Any] = {
    "method": "engle-granger",  # or "johansen"
    "max_pairs": 3,
    "pvalue": 0.05,
    "max_candidates": 40,
    "bonferroni": True,
    "entry_z": 2.0,
    "exit_z": 0.5,
    "stop_z": 4.0,
    "gross": 1.0,
}


class PairsCointegration(Strategy):
    """Trade z-score bands on cointegrated pairs with a hedge ratio frozen at fit."""

    name = "pairs-coint"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {**PAIR_DEFAULTS, "z_window": 63}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        if not 0 <= p["exit_z"] < p["entry_z"] < p["stop_z"]:
            raise ValueError("need 0 <= exit_z < entry_z < stop_z")
        self.pairs_: list[Pair] = []

    def fit(self, data: pd.DataFrame) -> Self:
        p = self.params
        log_close = np.log(to_wide(data, "close"))
        self.pairs_ = select_pairs(
            log_close,
            p["method"],
            p["max_pairs"],
            p["pvalue"],
            p["max_candidates"],
            p["bonferroni"],
        )
        self.is_fitted = True
        return self

    def spreads(self, data: pd.DataFrame) -> pd.DataFrame:
        log_close = np.log(to_wide(data, "close"))
        return pd.DataFrame(
            {
                f"{q.y}/{q.x}": log_close[q.y] - q.intercept - q.hedge * log_close[q.x]
                for q in self.pairs_
            },
            index=log_close.index,
        )

    def zscores(self, data: pd.DataFrame) -> pd.DataFrame:
        s = self.spreads(data)
        w = self.params["z_window"]
        roll = s.rolling(w, min_periods=w)
        return (s - roll.mean()) / roll.std()

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        p = self.params
        close = to_wide(data, "close")
        z = self.zscores(data)
        legs = []
        for q in self.pairs_:
            pos = band_positions(
                z[f"{q.y}/{q.x}"].to_numpy(), p["entry_z"], p["exit_z"], p["stop_z"]
            )
            legs.append((q.y, q.x, pos, np.full(len(pos), q.hedge)))
        return spread_weights(close.index, close.columns, legs, p["gross"])
