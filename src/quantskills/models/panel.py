"""Model fitting and time-aware cross-validation on (date, ticker) panels.

The CV score is the **mean daily Spearman IC** between predictions and labels in
each test fold: for a ranking strategy, getting the order right matters more
than getting the level right.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from quantskills.utils.splits import purged_kfold


def dropna_xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align features and label; keep rows where both are complete."""
    df = X.join(y, how="inner")
    df = df.dropna()
    return df[X.columns], df[y.name]


def panel_ic(pred: pd.Series, y: pd.Series, min_obs: int = 5) -> float:
    """Mean over dates of the cross-sectional Spearman correlation."""
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    ics = []
    for _, g in df.groupby(level="date"):
        if len(g) >= min_obs and g["p"].nunique() > 1 and g["y"].nunique() > 1:
            ics.append(g["p"].rank().corr(g["y"].rank()))
    return float(np.mean(ics)) if ics else float("nan")


def cv_score(
    make_model: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 4,
    horizon: int = 21,
    embargo: int = 5,
    predict: Callable[[Any, pd.DataFrame], np.ndarray] | None = None,
) -> list[float]:
    """Per-fold test IC under purged, embargoed k-fold (never shuffled)."""
    predict = predict or (lambda m, x: m.predict(x))
    dates = X.index.get_level_values("date")
    scores = []
    for train_d, test_d in purged_kfold(dates, n_splits, horizon, embargo):
        tr, te = dates.isin(train_d), dates.isin(test_d)
        model = make_model().fit(X[tr], y[tr])
        pred = pd.Series(predict(model, X[te]), index=X.index[te])
        scores.append(panel_ic(pred, y[te]))
    return scores
