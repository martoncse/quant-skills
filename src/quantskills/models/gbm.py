"""LightGBM helpers: boosting rounds chosen by purged CV, and SHAP importances.

SHAP values come from LightGBM's built-in TreeSHAP (``pred_contrib=True``), so the
separate ``shap`` package isn't needed.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from quantskills.models.panel import panel_ic
from quantskills.utils.splits import purged_kfold

DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 2,
    "deterministic": True,
    "force_row_wise": True,
}


def fit_lgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any] | None = None,
    max_rounds: int = 300,
    n_splits: int = 4,
    horizon: int = 21,
    embargo: int = 5,
    seed: int = 0,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Pick the number of rounds by early stopping in each purged fold, then refit on all data.

    Returns the final booster and a CV report (per-fold IC and best rounds).
    """
    p = {**DEFAULT_LGB_PARAMS, **(params or {}), "seed": seed}
    dates = X.index.get_level_values("date")
    best, ics = [], []
    for train_d, test_d in purged_kfold(dates, n_splits, horizon, embargo):
        tr, te = dates.isin(train_d), dates.isin(test_d)
        dtrain = lgb.Dataset(X[tr], y[tr])
        dvalid = lgb.Dataset(X[te], y[te], reference=dtrain)
        booster = lgb.train(
            p,
            dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        best.append(max(booster.best_iteration, 1))
        pred = pd.Series(booster.predict(X[te], num_iteration=booster.best_iteration), X.index[te])
        ics.append(panel_ic(pred, y[te]))
    rounds = int(np.median(best))
    final = lgb.train(p, lgb.Dataset(X, y), num_boost_round=rounds)
    return final, {"fold_ic": ics, "fold_rounds": best, "rounds": rounds}


def shap_importance(booster: lgb.Booster, X: pd.DataFrame, max_rows: int = 5000, seed: int = 0):
    """Mean |SHAP value| per feature, largest first."""
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    contrib = np.asarray(booster.predict(X, pred_contrib=True))[:, :-1]  # last column is the bias
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
