"""Strategies #2-#7: interface, no look-ahead, fit sees training data only, behaviour."""

import numpy as np
import pandas as pd
import pytest

from quantskills.data import PRICE_COLUMNS, synthetic_prices, to_wide
from quantskills.eval import tear_sheet
from quantskills.strategies import (
    STRATEGIES,
    BayesianPairs,
    GBMLongShort,
    LinearForecast,
    PairsCointegration,
    UnsupervisedRisk,
    VolatilityTiming,
    run_strategy,
)
from quantskills.strategies._pairs import band_positions
from quantskills.strategies.bayes_pairs import kalman_hedge
from quantskills.utils import assert_no_lookahead, holdout_date

PHASE3 = [
    LinearForecast,
    PairsCointegration,
    VolatilityTiming,
    BayesianPairs,
    GBMLongShort,
    UnsupervisedRisk,
]
FAST = {
    GBMLongShort: {"max_rounds": 40},
    UnsupervisedRisk: {"lookback": 126},
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return synthetic_prices(n_assets=20, n_days=800, seed=3, n_pairs=2, garch_index=True)


@pytest.fixture(scope="module")
def train_end(panel) -> pd.Timestamp:
    return holdout_date(panel.index.get_level_values("date"), 0.3)


def _long_from_close(close: pd.DataFrame) -> pd.DataFrame:
    s = close.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return pd.DataFrame({c: s for c in PRICE_COLUMNS}).sort_index()


def _train(panel, train_end):
    return panel[panel.index.get_level_values("date") <= train_end]


def test_registry_has_book_order():
    assert list(STRATEGIES) == [
        "factor-long-short",
        "linear-forecast",
        "pairs-coint",
        "vol-timing",
        "bayes-pairs",
        "gbm-long-short",
        "unsupervised-risk",
        "text-sentiment",
        "dl-forecast",
        "cond-autoencoder",
        "rl-dqn",
    ]
    for cls in STRATEGIES.values():
        assert (cls.__doc__ or "").strip(), f"{cls.name} needs a one-line docstring for the lab"


@pytest.mark.parametrize("cls", PHASE3, ids=lambda c: c.name)
def test_interface(cls, panel):
    with pytest.raises(TypeError):
        cls(not_a_param=1)
    if cls is not UnsupervisedRisk:  # nothing to learn, fit is a no-op
        with pytest.raises(RuntimeError):
            cls().signals(panel)


@pytest.mark.parametrize("cls", PHASE3, ids=lambda c: c.name)
def test_signals_have_no_lookahead(cls, panel, train_end):
    strat = cls(**FAST.get(cls, {})).fit(_train(panel, train_end))
    close = to_wide(panel, "close")
    cut = close.index.get_loc(train_end)
    assert_no_lookahead(
        lambda c: strat.signals(_long_from_close(c)), close, cut_points=[cut - 60, cut + 40]
    )


def _fitted_state(strat):
    for attr in ("pairs_", "coef_", "garch_params_", "shap_importance_", "obs_var_"):
        if hasattr(strat, attr):
            v = getattr(strat, attr)
            return v.round(12).to_dict() if isinstance(v, pd.Series) else v
    return None


@pytest.mark.parametrize("cls", PHASE3, ids=lambda c: c.name)
def test_fit_ignores_the_holdout(cls, panel, train_end):
    dates = panel.index.get_level_values("date")
    scrambled = panel.copy()
    future = dates > train_end
    scrambled.loc[future, "close"] *= np.random.default_rng(1).lognormal(0, 0.3, future.sum())
    a = cls(**FAST.get(cls, {})).fit(_train(panel, train_end))
    b = cls(**FAST.get(cls, {})).fit(scrambled[dates <= train_end])
    assert _fitted_state(a) == _fitted_state(b)


# --- #2 linear ---------------------------------------------------------------


@pytest.mark.parametrize("model", ["ridge", "lasso", "logistic"])
def test_linear_forecast_models(model, panel, train_end):
    s = LinearForecast(model=model).fit(_train(panel, train_end))
    assert s.alpha_ in s.cv_report_
    assert list(s.coef_.index) == list(s.params["features"])
    w = s.signals(panel).dropna(how="all")
    active = w[w.abs().sum(axis=1) > 0]
    np.testing.assert_allclose(active.sum(axis=1), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        LinearForecast(model="svm")


# --- #3 / #5 pairs -------------------------------------------------------------


def test_pairs_finds_the_planted_pairs(panel, train_end):
    s = PairsCointegration().fit(_train(panel, train_end))
    found = {frozenset((p.y, p.x)) for p in s.pairs_}
    assert {frozenset(("PAIR0A", "PAIR0B")), frozenset(("PAIR1A", "PAIR1B"))} <= found
    s_j = PairsCointegration(method="johansen").fit(_train(panel, train_end))
    assert frozenset(("PAIR0A", "PAIR0B")) in {frozenset((p.y, p.x)) for p in s_j.pairs_}


def test_pairs_trade_planted_pairs_profitably_out_of_sample(panel, train_end):
    strat = PairsCointegration()
    run = run_strategy(strat, panel, train_end)
    assert tear_sheet(run.out_of_sample)["sharpe"] > 0.5
    target = strat.signals(panel).dropna(how="all")
    assert (target.abs().sum(axis=1) <= 1.0 + 1e-9).all()
    # trades only when a position changes; other days are "hold" (NaN)
    assert len(target) < 0.5 * panel.index.get_level_values("date").nunique()


def test_band_positions_hysteresis():
    z = np.array([0, 2.5, 1.0, 0.4, -2.5, -1.0, -0.2, 5.0, np.nan, 3.0])
    pos = band_positions(z, entry=2.0, exit_=0.5, stop=4.0)
    np.testing.assert_array_equal(pos, [0, -1, -1, 0, 1, 1, 0, 0, 0, -1])


def test_kalman_tracks_a_moving_hedge_ratio():
    rng = np.random.default_rng(0)
    n = 1500
    x = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    h = np.linspace(0.8, 1.4, n)
    y = 0.2 + h * x + rng.normal(0, 0.01, n)
    _, est, z = kalman_hedge(y, x, prior=(0.2, 0.8), obs_var=1e-4, delta=1e-2)
    assert abs(est[-1] - 1.4) < 0.1
    assert 0.5 < np.nanstd(z) < 2.0


def test_bayes_pairs_rejects_bad_delta():
    with pytest.raises(ValueError):
        BayesianPairs(delta=0)


# --- #4 vol timing -------------------------------------------------------------


def test_vol_timing_hits_target_vol(panel, train_end):
    s = VolatilityTiming(target_vol=0.10)
    run = run_strategy(s, panel, train_end)
    assert s.ticker_ == "IDX"
    g = s.garch_params_
    assert 0.5 < g["alpha"] + g["beta"] < 1.0
    vol = tear_sheet(run.out_of_sample)["ann_vol"]
    assert 0.06 < vol < 0.14
    w = s.signals(panel).dropna(how="all")
    assert (w.drop(columns="IDX") == 0).all().all()
    assert w["IDX"].max() <= s.params["max_leverage"]


def test_vol_timing_ticker_param(panel, train_end):
    s = VolatilityTiming(ticker="syn01").fit(_train(panel, train_end))
    assert s.ticker_ == "SYN01"
    with pytest.raises(KeyError):
        VolatilityTiming(ticker="NOPE").fit(panel)


# --- #6 gradient boosting --------------------------------------------------------


def test_gbm_reports_cv_and_shap(panel, train_end):
    s = GBMLongShort(max_rounds=40).fit(_train(panel, train_end))
    assert s.cv_report_["rounds"] >= 1 and len(s.cv_report_["fold_ic"]) == 4
    assert set(s.shap_importance_.index) == set(s.params["features"])
    assert (s.shap_importance_ >= 0).all()


# --- #7 unsupervised ---------------------------------------------------------------


@pytest.mark.parametrize("method", ["hrp", "eigen", "min_var", "inverse_vol", "equal"])
def test_unsupervised_weights(method, panel):
    w = UnsupervisedRisk(method=method, lookback=126).fit(panel).signals(panel)
    rebal = w.dropna(how="all").iloc[1:]
    rebal = rebal[rebal.abs().sum(axis=1) > 0]
    assert len(rebal) > 10
    if method == "eigen":
        np.testing.assert_allclose(rebal.abs().sum(axis=1), 1.0, atol=1e-9)
    else:
        np.testing.assert_allclose(rebal.sum(axis=1), 1.0, atol=1e-6)
        assert (rebal >= -1e-12).all().all()
