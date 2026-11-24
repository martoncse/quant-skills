import numpy as np
import pandas as pd
import pytest

from quantskills.backtest import CostModel, run_backtest
from quantskills.data import synthetic_prices, to_wide
from quantskills.eval import sharpe_ratio
from quantskills.robustness import (
    add_price_noise,
    delay_test,
    delay_weights,
    noise_test,
    regime_labels,
    regime_table,
    sensitivity_surface,
)
from quantskills.robustness.perturb import walk_forward_backtest
from quantskills.strategies import FactorLongShort
from quantskills.utils import assert_no_lookahead


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_prices(n_assets=20, n_days=700, seed=3)


def test_price_noise_keeps_bars_consistent(prices):
    noisy = add_price_noise(prices, 50, seed=1)
    assert (noisy["low"] <= noisy["close"] + 1e-12).all()
    assert (noisy["close"] <= noisy["high"] + 1e-12).all()
    pd.testing.assert_series_equal(noisy["volume"], prices["volume"])
    ratio = np.log(noisy["close"] / prices["close"])
    assert ratio.std() == pytest.approx(50e-4, rel=0.1)
    pd.testing.assert_frame_equal(add_price_noise(prices, 50, 1), noisy)  # seeded
    pd.testing.assert_frame_equal(add_price_noise(prices, 0), prices)
    with pytest.raises(ValueError):
        add_price_noise(prices, -1)


def test_delay_shifts_later_only():
    w = pd.DataFrame({"A": [1.0, np.nan, 0.5]}, index=pd.bdate_range("2020-01-01", periods=3))
    d = delay_weights(w, 1)["A"]
    assert np.isnan(d.iloc[0]) and d.iloc[1] == 1.0 and np.isnan(d.iloc[2])
    with pytest.raises(ValueError, match="look-ahead"):
        delay_weights(w, -1)


def test_delay_zero_matches_plain_backtest(prices):
    strat = FactorLongShort().fit(prices)
    w, close = strat.signals(prices), to_wide(prices, "close")
    out = delay_test(w, close, (0, 1))
    plain = sharpe_ratio(run_backtest(w, close).trim_warmup().returns)
    assert out.loc[out.delay == 0, "sharpe"].item() == pytest.approx(plain)


def test_walk_forward_folds_only_use_the_past(prices):
    """Scrambling prices after fold 1 must not change fold-1 returns."""
    res, _ = walk_forward_backtest(FactorLongShort, {}, prices, n_folds=3)
    dates = to_wide(prices, "close").index
    fold1_end = res.returns.index[len(res.returns) // 3 - 1]
    later = prices.index.get_level_values("date") > fold1_end
    scrambled = prices.copy()
    rng = np.random.default_rng(0)
    factor = rng.lognormal(0, 0.3, later.sum())
    for col in ("open", "high", "low", "close"):
        scrambled.loc[later, col] = scrambled.loc[later, col].to_numpy() * factor
    res2, _ = walk_forward_backtest(FactorLongShort, {}, scrambled, n_folds=3)
    a = res.returns.loc[:fold1_end]
    b = res2.returns.loc[:fold1_end]
    assert len(a) > 50 and dates[0] < a.index[0]
    pd.testing.assert_series_equal(a, b)


def test_noise_and_surface_shapes(prices):
    noise = noise_test(FactorLongShort, {}, prices, (10.0,), (0, 1), mode="in-sample")
    assert list(noise.columns) == ["bps", "seed", "sharpe"] and len(noise) == 2
    surface, trials = sensitivity_surface(
        FactorLongShort,
        {},
        {"quantile": (0.1, 0.2, 0.7), "rebalance_days": (10, 21)},
        prices,
        CostModel(),
        mode="in-sample",
    )
    assert surface.shape == (3, 2) and surface.index.name == "quantile"
    assert surface.loc[0.7].isna().all()  # invalid quantile: skipped, not crashed
    assert trials.shape[1] == 4 and not trials.isna().any().any()


def test_surface_needs_two_axes(prices):
    with pytest.raises(ValueError, match="two"):
        sensitivity_surface(FactorLongShort, {}, {"quantile": (0.1,)}, prices)


def _codes(close: pd.DataFrame, kind: str) -> pd.DataFrame:
    lab = regime_labels(close, kind)
    return lab.map({"high vol": 1.0, "low vol": 0.0, "up trend": 1.0, "down trend": 0.0}).to_frame()


@pytest.mark.parametrize("kind", ["vol", "trend"])
def test_regime_labels_no_lookahead(prices, kind):
    close = to_wide(prices, "close")
    assert_no_lookahead(lambda c: _codes(c, kind), close)


def test_regime_labels_values(prices):
    close = to_wide(prices, "close")
    vol = regime_labels(close, "vol")
    assert set(vol.dropna()) == {"high vol", "low vol"}
    assert vol.iloc[:62].isna().all()
    assert regime_labels(close, "trend").iloc[:199].isna().all()
    with pytest.raises(ValueError):
        regime_labels(close, "moon")


def test_regime_table_lags_labels():
    idx = pd.bdate_range("2020-01-01", periods=200)
    labels = pd.Series(["a", "b"] * 100, index=idx)
    # Return on day t is +1% exactly when day t-1 was labelled "a".
    r = pd.Series(np.where(labels.shift(1) == "a", 0.01, -0.01), index=idx)
    t = regime_table(r, labels, min_days=20)
    assert t.loc["a", "ann_return"] > 0 > t.loc["b", "ann_return"]
    assert t["days"].sum() == 199
    assert regime_table(r, labels, min_days=500)["sharpe"].isna().all()
