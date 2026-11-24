"""Strategy #6: gradient-boosted long-short (LightGBM + SHAP, purged CV).

Hypothesis: trees can pick up non-linear interactions between price features
(e.g. momentum that only works in low-volatility names) that a linear model misses.

Method:
    1. Same features and rank label as strategy #2.
    2. LightGBM regression. The number of boosting rounds is set by early stopping
       inside each **purged, embargoed** CV fold (median best round), then the model
       is refit on the whole training window. Shallow trees and large leaves
       (``min_data_in_leaf``) keep it from memorising noise.
    3. SHAP values (TreeSHAP, built into LightGBM) explain which features drive the
       forecast; see ``shap_importance_``.
    4. Long the top, short the bottom quantile.

Book reference: ML4T 3rd ed., ch. 12 (gradient boosting, SHAP) and ch. 6/7 (purged CV).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantskills.models.gbm import fit_lgbm_cv, shap_importance
from quantskills.strategies._panel_ml import PANEL_DEFAULTS, PanelMLStrategy


class GBMLongShort(PanelMLStrategy):
    """LightGBM forecast of peer-relative returns with purged CV, traded long-short."""

    name = "gbm-long-short"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            **PANEL_DEFAULTS,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 200,
            "max_rounds": 300,
            "seed": 0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.shap_importance_: pd.Series = pd.Series(dtype=float)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        p = self.params
        lgb_params = {
            "learning_rate": p["learning_rate"],
            "num_leaves": p["num_leaves"],
            "min_data_in_leaf": p["min_data_in_leaf"],
        }
        self.booster_, self.cv_report_ = fit_lgbm_cv(
            X,
            y,
            lgb_params,
            max_rounds=p["max_rounds"],
            n_splits=p["cv_splits"],
            horizon=p["horizon"],
            embargo=p["embargo"],
            seed=p["seed"],
        )
        self.shap_importance_ = shap_importance(self.booster_, X, seed=p["seed"])

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster_.predict(X))
