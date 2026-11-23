import numpy as np
import pandas as pd
import pytest

from quantskills.backtest import CostModel, run_backtest


def _prices():
    idx = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0], "B": [50.0, 50.0, 55.0, 55.0]}, idx)


def test_costs_are_on_by_default():
    assert CostModel().rate > 0


def test_hand_computed_single_asset():
    close = _prices()[["A"]]
    w = pd.DataFrame({"A": [1.0, np.nan, np.nan, 0.0]}, index=close.index)
    res = run_backtest(w, close, CostModel(10, 0))  # 10 bps
    # day0: buy 1.0 -> cost 0.001; day1: +10%; day2: -10%; day3: sell -> cost 0.001
    np.testing.assert_allclose(res.gross_returns.to_numpy(), [0.0, 0.10, -0.10, 0.0])
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(res.returns.to_numpy(), [-0.001, 0.10, -0.10, -0.001])


def test_weights_earn_next_period_returns_only():
    """A same-day 'perfect foresight' weight must not capture that day's return."""
    close = _prices()
    rets = close.pct_change()
    cheat = np.sign(rets).fillna(0.0)  # knows today's return at today's close
    res = run_backtest(cheat, close, CostModel(0, 0))
    expected = (cheat.shift(1).fillna(0) * rets.fillna(0)).sum(axis=1)
    np.testing.assert_allclose(res.gross_returns.to_numpy(), expected.to_numpy())


def test_weights_drift_between_rebalances():
    close = _prices()
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    w.iloc[0] = [0.5, 0.5]
    res = run_backtest(w, close, CostModel(0, 0))
    # after day1, A rose 10% and B flat -> A weight 0.55/1.05
    assert res.weights.iloc[1]["A"] == pytest.approx(0.55 / 1.05)
    # buy-and-hold: total return equals portfolio value change
    total = (1 + res.returns).prod() - 1
    assert total == pytest.approx(0.5 * (99 / 100) + 0.5 * (55 / 50) - 1)
    assert res.turnover.iloc[1:].sum() == 0


def test_rebalancing_to_same_target_charges_drift_turnover():
    close = _prices()
    w = pd.DataFrame([[0.5, 0.5]] * 4, index=close.index, columns=close.columns)
    res = run_backtest(w, close)
    assert res.turnover.iloc[1] > 0
    assert (res.returns <= res.gross_returns).all()


def test_split():
    close = _prices()
    w = pd.DataFrame([[0.5, 0.5]] * 4, index=close.index, columns=close.columns)
    a, b = run_backtest(w, close).split(close.index[1])
    assert len(a.returns) == 2 and len(b.returns) == 2


def test_trim_warmup_drops_leading_flat_days():
    close = _prices()
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    w.iloc[0] = 0.0
    w.iloc[2] = [0.5, 0.5]
    trimmed = run_backtest(w, close).trim_warmup()
    assert trimmed.returns.index[0] == close.index[2]
