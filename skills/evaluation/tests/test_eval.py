import numpy as np
import pandas as pd
import pytest

from quantskills.backtest import run_backtest
from quantskills.eval import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    format_tear_sheet,
    ic_summary,
    information_coefficient,
    max_drawdown,
    probabilistic_sharpe_ratio,
    quantile_returns,
    sharpe_ratio,
    summary,
    tear_sheet,
)
from quantskills.features import forward_returns


def _noise(n=1000, mu=0.0005, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.bdate_range("2020-01-01", periods=n))


def test_sharpe_matches_formula():
    r = _noise()
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))


def test_max_drawdown_known_path():
    r = pd.Series([0.0, 0.5, -0.5, 0.2])  # 1 -> 1.5 -> 0.75 -> 0.9
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_summary_keys():
    s = summary(_noise())
    assert {"ann_return", "sharpe", "max_drawdown", "hit_rate"} <= set(s)


def test_psr_increases_with_skill():
    assert probabilistic_sharpe_ratio(_noise(mu=0.001)) > probabilistic_sharpe_ratio(_noise(mu=0.0))


def test_expected_max_sharpe_grows_with_trials():
    vals = [expected_max_sharpe(n, 0.01) for n in (1, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert all(a < b for a, b in zip(vals, vals[1:], strict=False))


def test_dsr_penalises_many_trials():
    r = _noise(mu=0.0006)
    one = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=500)
    assert one.dsr == pytest.approx(one.psr)
    assert many.dsr < one.dsr
    assert many.sharpe_threshold > 0


def test_dsr_uses_trial_dispersion():
    r = _noise(mu=0.0006)
    tight = deflated_sharpe_ratio(r, trial_sharpes=[0.9, 1.0, 1.1])
    wide = deflated_sharpe_ratio(r, trial_sharpes=[-1.0, 1.0, 3.0])
    assert wide.dsr < tight.dsr


def test_ic_perfect_and_quantiles_monotone(close):
    fwd = forward_returns(close, 5)
    ic = information_coefficient(fwd, fwd)
    assert np.allclose(ic, 1.0)
    q = quantile_returns(fwd, fwd, 5)
    assert list(q.index) == [1, 2, 3, 4, 5]
    assert q.is_monotonic_increasing
    assert ic_summary(ic, horizon=5)["mean_ic"] == pytest.approx(1.0)


def test_ic_noise_factor_is_near_zero(close):
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=close.shape), index=close.index, columns=close.columns)
    ic = information_coefficient(noise, forward_returns(close, 1))
    assert abs(ic.mean()) < 0.03


def test_tear_sheet(close):
    w = pd.DataFrame(1.0 / close.shape[1], index=close.index, columns=close.columns)
    sheet = tear_sheet(run_backtest(w, close), n_trials=3)
    assert sheet["n_trials"] == 3
    assert sheet["ann_cost_drag"] > 0
    assert "not a forecast" in format_tear_sheet(sheet)
