"""Shared machinery for cross-sectional ML strategies (#2 linear, #6 gradient boosting).

``fit`` builds features and labels from the training window only. Labels for
the last ``horizon`` training dates are NaN (their outcome lies past the end
of the window) and are dropped, so nothing leaks from the hold-out. ``signals``
scores every date with the frozen model and trades the quantile long-short book.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.features.panel import feature_panel, from_panel, label_panel
from quantskills.models.panel import dropna_xy
from quantskills.strategies._weights import quantile_long_short
from quantskills.strategies.base import Strategy

PANEL_DEFAULTS: dict[str, Any] = {
    "features": ("mom_12_1", "mom_6_1", "rev_5", "rev_21", "lowvol_21", "lowvol_63", "dist_sma50"),
    "horizon": 21,
    "rebalance_days": 21,
    "quantile": 0.2,
    "gross": 1.0,
    "min_assets": 10,
    "cv_splits": 4,
    "embargo": 5,
}


class PanelMLStrategy(Strategy):
    label_kind = "rank"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if not 0 < self.params["quantile"] <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        self.cv_report_: dict[Any, Any] = {}

    def _features(self, data: pd.DataFrame) -> pd.DataFrame:
        return feature_panel(to_wide(data, "close"), self.params["features"])

    def fit(self, data: pd.DataFrame) -> Self:
        close = to_wide(data, "close")
        X = feature_panel(close, self.params["features"])
        y = label_panel(close, self.params["horizon"], self.label_kind)
        X, y = dropna_xy(X, y)
        if len(X) < 500:
            raise ValueError(f"only {len(X)} complete training rows; need more history")
        self._fit_model(X, y)
        self.is_fitted = True
        return self

    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Model prediction per (date, ticker), wide; NaN where any feature is missing."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        X = self._features(data)
        complete = X.notna().all(axis=1)
        pred = pd.Series(np.nan, index=X.index)
        if complete.any():
            pred[complete] = self._predict(X[complete])
        wide = from_panel(pred)
        return wide.reindex(index=to_wide(data, "close").index)

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return quantile_long_short(
            self.score(data), p["quantile"], p["gross"], p["rebalance_days"], p["min_assets"]
        )

    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray: ...
