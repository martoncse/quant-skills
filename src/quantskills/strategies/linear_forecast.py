"""Strategy #2: linear return forecast (ridge / lasso / logistic direction).

Hypothesis: a linear combination of price features predicts which stocks will
beat their peers over the next ``horizon`` days.

Method:
    1. Features: cross-sectional z-scores of momentum, reversal, volatility and
       trend (``features.panel``), past data only.
    2. Label: the cross-sectional rank of the forward return (regression), or
       whether it beats the cross-sectional median (logistic "direction" model).
       Ranking against peers removes the market move, which a long-short book
       doesn't trade.
    3. Regularisation strength is chosen by **purged, embargoed k-fold CV** on the
       training window, scored by mean daily IC. Random K-fold would leak because
       neighbouring days share overlapping labels.
    4. Long the top, short the bottom quantile of the prediction.

Book reference: ML4T 3rd ed., ch. 11 (the ML pipeline: linear models) and ch. 7 (labels, IC).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

from quantskills.models.panel import cv_score
from quantskills.strategies._panel_ml import PANEL_DEFAULTS, PanelMLStrategy

MODELS = ("ridge", "lasso", "logistic")
DEFAULT_ALPHAS = {
    "ridge": (1.0, 100.0, 10_000.0),
    "lasso": (1e-5, 1e-4, 1e-3),
    "logistic": (1.0, 100.0, 10_000.0),  # alpha = 1 / C
}


class LinearForecast(PanelMLStrategy):
    """Ridge, lasso or logistic forecast of peer-relative returns, traded long-short."""

    name = "linear-forecast"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"model": "ridge", "alphas": (), **PANEL_DEFAULTS}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["model"] not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.alpha_: float = float("nan")
        self.coef_: pd.Series = pd.Series(dtype=float)

    @property
    def label_kind(self) -> str:  # type: ignore[override]
        return "up" if self.params["model"] == "logistic" else "rank"

    def _make(self, alpha: float):
        kind = self.params["model"]
        if kind == "ridge":
            return Ridge(alpha=alpha)
        if kind == "lasso":
            return Lasso(alpha=alpha, max_iter=5000)
        return LogisticRegression(C=1.0 / alpha, max_iter=1000)

    def _raw_predict(self, model, X: pd.DataFrame) -> np.ndarray:
        if self.params["model"] == "logistic":
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        p = self.params
        alphas = tuple(p["alphas"]) or DEFAULT_ALPHAS[p["model"]]
        report = {}
        for a in alphas:
            folds = cv_score(
                partial(self._make, a),
                X,
                y,
                p["cv_splits"],
                p["horizon"],
                p["embargo"],
                predict=self._raw_predict,
            )
            report[float(a)] = float(np.nanmean(folds))
        self.cv_report_ = report
        self.alpha_ = max(report, key=lambda a: -np.inf if np.isnan(report[a]) else report[a])
        self.model_ = self._make(self.alpha_).fit(X, y)
        coef = np.ravel(self.model_.coef_)
        self.coef_ = pd.Series(coef, index=X.columns)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._raw_predict(self.model_, X)
