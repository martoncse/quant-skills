import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from quantskills.data import to_wide
from quantskills.features import feature_panel, label_panel
from quantskills.models import (
    cv_score,
    dropna_xy,
    fit_lgbm_cv,
    oof_predictions,
    panel_ic,
    shap_importance,
)
from quantskills.utils import purged_kfold, walk_forward_splits

DATES = pd.bdate_range("2020-01-01", periods=200)


def test_purged_kfold_never_overlaps_label_windows():
    h, emb = 10, 3
    folds = list(purged_kfold(DATES, n_splits=5, horizon=h, embargo=emb))
    assert len(folds) == 5
    covered = pd.DatetimeIndex([])
    for train, test in folds:
        assert train.intersection(test).empty
        pos = DATES.get_indexer(train)
        lo, hi = DATES.get_loc(test[0]), DATES.get_loc(test[-1])
        # no training date within `h` before the test block or `emb` after it
        assert not ((pos >= lo - h) & (pos <= hi + emb)).any()
        covered = covered.union(test)
    assert covered.equals(DATES)  # each date is tested exactly once


def test_walk_forward_trains_only_on_the_past():
    folds = list(walk_forward_splits(DATES, n_splits=4, horizon=5))
    assert len(folds) == 4
    for train, test in folds:
        assert train.max() < test.min()
        gap = DATES.get_loc(test.min()) - DATES.get_loc(train.max())
        assert gap == 6  # horizon 5 purged + 1
    assert folds[-1][1][-1] == DATES[-1]


def test_splitters_reject_bad_input():
    with pytest.raises(ValueError):
        list(purged_kfold(DATES[:5], n_splits=5))
    with pytest.raises(ValueError):
        list(purged_kfold(DATES, n_splits=1))
    with pytest.raises(ValueError):
        list(walk_forward_splits(DATES[:3], n_splits=5, horizon=5))


def test_panel_ic_perfect_and_inverse():
    idx = pd.MultiIndex.from_product([DATES[:20], list("abcdefg")], names=["date", "ticker"])
    y = pd.Series(np.random.default_rng(0).normal(size=len(idx)), idx)
    assert panel_ic(y, y) == pytest.approx(1.0)
    assert panel_ic(-y, y) == pytest.approx(-1.0)


@pytest.fixture(scope="module")
def xy(prices):
    close = to_wide(prices, "close")
    return dropna_xy(feature_panel(close), label_panel(close, 21))


def test_cv_score_returns_one_ic_per_fold(xy):
    X, y = xy
    scores = cv_score(lambda: Ridge(alpha=10.0), X, y, n_splits=3, horizon=21)
    assert len(scores) == 3 and all(-1 <= s <= 1 for s in scores)


def test_lgbm_cv_and_shap(xy):
    X, y = xy
    booster, report = fit_lgbm_cv(X, y, max_rounds=30, n_splits=3)
    assert 1 <= report["rounds"] <= 30
    imp = shap_importance(booster, X, max_rows=2000)
    assert list(imp.index) != [] and (imp >= 0).all()
    # SHAP contributions + bias reproduce the prediction exactly
    contrib = booster.predict(X.iloc[:50], pred_contrib=True)
    np.testing.assert_allclose(contrib.sum(axis=1), booster.predict(X.iloc[:50]), rtol=1e-6)


class _Recorder:
    """A model that predicts the latest training date it saw (as an ordinal)."""

    def fit(self, X, y):
        self.last_ = X.index.get_level_values("date").max().toordinal()
        return self

    def predict(self, X):
        return np.full(len(X), float(self.last_))


def test_oof_predictions_come_from_models_fit_on_the_past():
    idx = pd.MultiIndex.from_product([DATES, ["A", "B"]], names=["date", "ticker"])
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    y = pd.Series(0.0, index=idx, name="y")
    h = 10
    pred = oof_predictions(_Recorder, X, y, n_splits=3, horizon=h, min_train_frac=0.4)
    pred_dates = pred.index.get_level_values("date")
    assert pred_dates.min() >= DATES[80] and pred_dates.max() == DATES[-1]
    for d, last_fit in zip(pred_dates, pred.to_numpy(), strict=True):
        # the model that predicted day d stopped training more than h dates earlier
        assert DATES.get_loc(d) - DATES.get_loc(pd.Timestamp.fromordinal(int(last_fit))) > h
