import numpy as np
import pandas as pd
import pytest

from quantskills.data import PRICE_COLUMNS, synthetic_prices
from quantskills.strategies import (
    STRATEGIES,
    FactorLongShort,
    Strategy,
    get_strategy,
    run_strategy,
)
from quantskills.utils import assert_no_lookahead, holdout_date


def _long_from_close(close: pd.DataFrame) -> pd.DataFrame:
    s = close.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return pd.DataFrame({c: s for c in PRICE_COLUMNS}).sort_index()


def test_registry_and_interface():
    assert "factor-long-short" in STRATEGIES
    s = get_strategy("factor-long-short", quantile=0.1)
    assert isinstance(s, Strategy)
    assert s.params["quantile"] == 0.1
    with pytest.raises(TypeError):
        FactorLongShort(not_a_param=1)
    with pytest.raises(KeyError):
        get_strategy("nope")


def test_signals_require_fit(prices):
    with pytest.raises(RuntimeError):
        FactorLongShort().signals(prices)


def test_weights_are_dollar_neutral_with_target_gross(prices):
    w = FactorLongShort(gross=1.0).fit(prices).signals(prices)
    rebal = w.dropna(how="all")
    active = rebal[rebal.abs().sum(axis=1) > 0]
    assert len(active) > 5
    np.testing.assert_allclose(active.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(active.abs().sum(axis=1), 1.0, atol=1e-12)
    # rebalance every 21 days
    gaps = np.diff([w.index.get_loc(d) for d in active.index])
    assert set(gaps) == {21}


def test_signals_have_no_lookahead(prices, close):
    strat = FactorLongShort().fit(prices)
    cut = len(close) // 2
    assert_no_lookahead(
        lambda c: strat.signals(_long_from_close(c)), close, cut_points=[cut, cut + 30]
    )


def test_fit_uses_only_training_data(prices):
    dates = prices.index.get_level_values("date")
    train_end = holdout_date(dates, 0.3)
    train = prices[dates <= train_end]
    a = FactorLongShort().fit(train).factor_weights_
    # Scrambling the hold-out must not change fitted weights.
    scrambled = prices.copy()
    future = dates > train_end
    scrambled.loc[future, "close"] *= np.random.default_rng(0).lognormal(0, 0.2, future.sum())
    b = FactorLongShort().fit(scrambled[dates <= train_end]).factor_weights_
    assert a == b


def test_run_strategy_splits_in_and_out_of_sample(prices):
    dates = prices.index.get_level_values("date")
    train_end = holdout_date(dates, 0.3)
    run = run_strategy(FactorLongShort(), prices, train_end)
    assert run.in_sample.returns.index.max() <= train_end
    assert run.out_of_sample.returns.index.min() > train_end
    assert run.full.costs.sum() > 0
    with pytest.raises(ValueError):
        run_strategy(FactorLongShort(), prices, "1990-01-01")


def test_finds_planted_momentum_and_not_noise():
    """On data with planted momentum, momentum IC should be positive in training;
    on a pure-noise panel, the strategy's out-of-sample Sharpe should be modest."""
    planted = synthetic_prices(n_assets=40, n_days=1500, seed=11, drift_vol=0.001)
    s = FactorLongShort().fit(planted)
    assert s.train_ic_["momentum"] > 0

    null = synthetic_prices(n_assets=40, n_days=1500, seed=11, drift_vol=0.0)
    dates = null.index.get_level_values("date")
    run = run_strategy(FactorLongShort(), null, holdout_date(dates, 0.4))
    assert run.out_of_sample.returns.mean() * 252 < 0.1


def test_sweep_params_only_sees_training_data(prices):
    from quantskills.strategies import sweep_params

    dates = prices.index.get_level_values("date")
    train_end = holdout_date(dates, 0.3)
    grid = [{"quantile": q} for q in (0.1, 0.2, 0.3)]
    table = sweep_params(FactorLongShort, grid, prices, train_end)
    assert len(table) == 3 and table["is_sharpe"].notna().all()
    scrambled = prices.copy()
    future = dates > train_end
    scrambled.loc[future, "close"] *= 2.0
    again = sweep_params(FactorLongShort, grid, scrambled, train_end)
    pd.testing.assert_frame_equal(table, again)
