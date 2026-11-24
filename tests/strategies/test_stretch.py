"""Stretch goals: #11 synthetic histories for stress tests, #12 the RL trading agent."""

import numpy as np
import pandas as pd
import pytest

from quantskills.data import (
    PRICE_COLUMNS,
    stylized_prices,
    synthetic_prices,
    to_wide,
    validate_prices,
)
from quantskills.robustness.stress import (
    GENERATORS,
    bootstrap_paths,
    discriminative_score,
    gaussian_paths,
    predictive_score,
    returns_to_panel,
    stationary_bootstrap_index,
    stress_summary,
    stress_test,
    stylized_facts,
    synthetic_paths,
)
from quantskills.strategies import FactorLongShort, PairsCointegration, TextSentiment
from quantskills.utils import assert_no_lookahead, holdout_date


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return to_wide(stylized_prices(n_assets=4, n_days=600, seed=1)).pct_change().iloc[1:]


# --- #11 generators and fidelity ------------------------------------------------------


def test_stylized_panel_has_the_facts(returns):
    validate_prices(stylized_prices(n_assets=3, n_days=100))
    f = stylized_facts(returns)
    assert f["excess_kurtosis"] > 1.0  # fat tails
    assert f["acf_abs_1to5"] > 0.03  # volatility clustering
    assert abs(f["acf1_returns"]) < 0.1  # but returns themselves barely autocorrelated
    assert 0.3 < f["mean_pairwise_corr"] < 0.8


def test_gaussian_keeps_covariance_loses_tails(returns):
    paths = gaussian_paths(returns, 3, seed=0)
    assert all(p.shape == returns.shape and p.index.equals(returns.index) for p in paths)
    f = pd.DataFrame([stylized_facts(p) for p in paths]).mean()
    real = stylized_facts(returns)
    assert f["mean_pairwise_corr"] == pytest.approx(real["mean_pairwise_corr"], abs=0.08)
    assert abs(f["excess_kurtosis"]) < 0.5 < real["excess_kurtosis"]


def test_bootstrap_resamples_whole_days_in_blocks(returns):
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_index(1000, 20.0, rng)
    jumps = (np.diff(idx) != 1) & ~((idx[:-1] == 999) & (idx[1:] == 0))
    assert 1000 / (jumps.sum() + 1) == pytest.approx(20.0, rel=0.35)  # mean block length
    p = bootstrap_paths(returns, 2, seed=0)[0]
    rows = {tuple(np.round(r, 12)) for r in returns.to_numpy()}
    assert all(tuple(np.round(r, 12)) in rows for r in p.to_numpy())  # every day is a real day
    f = stylized_facts(p)
    assert f["excess_kurtosis"] > 1.0  # tails survive
    with pytest.raises(ValueError):
        synthetic_paths(returns, "dream", 1)
    assert set(GENERATORS) == {"gaussian", "bootstrap", "timegan"}


def test_returns_to_panel_round_trip(returns):
    panel = returns_to_panel(returns)
    validate_prices(panel)
    assert list(panel.columns) == list(PRICE_COLUMNS)
    back = to_wide(panel).pct_change().iloc[1:]
    pd.testing.assert_frame_equal(back, returns.iloc[1:], check_names=False, check_freq=False)


def test_fidelity_scores_separate_good_and_bad_fakes(returns):
    from quantskills.models.timegan import make_windows

    real = make_windows(returns.to_numpy(), 24, stride=24)
    same = make_windows(returns.to_numpy()[::-1].copy(), 24, stride=24)  # real days, reordered
    junk = real * 5.0  # far too volatile
    assert discriminative_score(real, junk) > 0.3
    assert discriminative_score(real, same) < discriminative_score(real, junk)
    half = len(returns) // 2
    tr = make_windows(returns.to_numpy()[:half], 24, stride=24)
    te = make_windows(returns.to_numpy()[half:], 24, stride=24)
    scores = predictive_score(tr, te, tr)
    assert scores["tstr_mae"] == pytest.approx(scores["trtr_mae"])


def test_make_windows_shapes():
    from quantskills.models.timegan import make_windows

    w = make_windows(np.arange(20.0).reshape(10, 2), 4, stride=2)
    assert w.shape == (4, 4, 2) and w[1, 0, 0] == 4.0
    with pytest.raises(ValueError):
        make_windows(np.zeros((3, 2)), 4)


def test_timegan_fits_and_samples_in_data_units(returns):
    pytest.importorskip("torch")
    from quantskills.models.timegan import TimeGAN, TimeGANConfig, make_windows

    cfg = TimeGANConfig(seq_len=12, hidden=8, embed_steps=5, supervised_steps=5, joint_steps=5)
    g = TimeGAN(cfg).fit(make_windows(returns.to_numpy(), 12))
    s1, s2 = g.sample(6, seed=1), g.sample(6, seed=1)
    assert s1.shape == (6, 12, 4) and np.isfinite(s1).all()
    np.testing.assert_array_equal(s1, s2)  # seeded
    assert np.abs(s1).max() < 1.0  # daily returns, not [0, 1] units
    assert len(g.log_.generator) == 5
    with pytest.raises(ValueError):
        TimeGAN(cfg).fit(np.zeros((5, 10, 2)))
    with pytest.raises(RuntimeError):
        TimeGAN(cfg).sample(1)
    from quantskills.robustness.stress import path_windows

    pw = path_windows(returns, 12)
    assert pw.shape[1:] == (12, 4)
    np.testing.assert_allclose(pw[0, 0], np.log1p(returns.to_numpy()[0]))
    g2 = TimeGAN(cfg).fit(pw)
    paths = synthetic_paths(returns, "timegan", 2, model=g2)
    assert paths[0].shape == returns.shape and np.isfinite(paths[0].to_numpy()).all()


# --- #11 stress test ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def plus() -> pd.DataFrame:
    return synthetic_prices(n_assets=20, n_days=800, seed=3, n_pairs=2)


def test_stress_test_shape_and_summary(plus):
    table = stress_test(FactorLongShort(), plus, "bootstrap", n_paths=3)
    assert list(table["path"]) == [-1, 0, 1, 2]
    s = stress_summary(table)
    assert s["n_paths"] == 3 and 0.0 <= s["is_sharpe_pct_real"] <= 1.0
    with pytest.raises(ValueError, match="news"):
        stress_test(TextSentiment(), plus, n_paths=1)


def test_bootstrap_breaks_cointegration(plus):
    """Resampling days keeps same-day comovement but not price levels: pairs find nothing."""
    table = stress_test(PairsCointegration(), plus, "bootstrap", n_paths=3)
    real = table[table["path"] == -1].iloc[0]
    fake = table[table["path"] >= 0]
    assert real["is_sharpe"] > 0.5
    assert (fake["is_sharpe"].abs() < 0.5).all()


# --- #12 RL ----------------------------------------------------------------------------------


def test_trend_state_features_no_lookahead():
    from quantskills.models.rl import scaled_return_features, trend_state_features

    close = to_wide(synthetic_prices(n_assets=3, n_days=300, seed=0))
    for fn in (trend_state_features, scaled_return_features):
        assert_no_lookahead(lambda c, fn=fn: pd.DataFrame(fn(c.iloc[:, 0]), index=c.index), close)
    f = trend_state_features(close.iloc[:, 0])
    assert f.shape == (300, 4) and np.isnan(f[:60]).all() and np.isfinite(f[100]).all()


def test_trading_core_reward_and_costs():
    from quantskills.models.rl import TradingCore

    X = np.zeros((20, 2))
    r = np.full(20, 0.01)
    env = TradingCore(X, r, cost_bps=10.0, episode_len=5, hold_days=2)
    env.reset(start=0)
    _, reward, done = env.step(2)  # go long: two days of +1%, pay 10 bps once
    assert reward == pytest.approx(100 * (0.02 - 0.001))
    _, reward, _ = env.step(2)  # stay long: no cost
    assert reward == pytest.approx(2.0)
    _, reward, _ = env.step(0)  # flip to short: pay for 2 units
    assert reward == pytest.approx(100 * (-0.02 - 0.002))
    with pytest.raises(ValueError):
        TradingCore(X, r, hold_days=0)
    with pytest.raises(ValueError, match="episode"):
        TradingCore(X[:3], r[:3], episode_len=5)


def test_gym_env_follows_the_api():
    pytest.importorskip("gymnasium")
    from quantskills.models.rl import make_env

    env = make_env(np.zeros((50, 3)), np.full(50, 0.001), episode_len=10)
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,) and env.action_space.n == 3 and info == {}
    obs, reward, terminated, truncated, _ = env.step(2)
    assert obs[-1] == 1.0 and not truncated


RL_FAST = {"train_steps": 600, "eval_every": 300, "hidden": 16}


@pytest.fixture(scope="module")
def rl_fit(plus):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from quantskills.strategies import RLAgent

    te = holdout_date(plus.index.get_level_values("date"), 0.3)
    train = plus[plus.index.get_level_values("date") <= te]
    return RLAgent(**RL_FAST).fit(train), te


def test_rl_agent_trades_one_asset_without_lookahead(rl_fit, plus):
    agent, te = rl_fit
    w = agent.signals(plus)
    assert set(np.unique(w[agent.ticker_])) <= {-1.0, 0.0, 1.0}
    assert (w.drop(columns=agent.ticker_) == 0).all().all()
    # weekly decisions: positions change at most every `hold_days` rows
    changes = np.flatnonzero(np.diff(w[agent.ticker_].to_numpy()) != 0)
    assert (np.diff(changes) >= agent.params["hold_days"]).all()
    close = to_wide(plus, "close")
    cut = close.index.get_loc(te)

    def as_long(c: pd.DataFrame) -> pd.DataFrame:
        s = c.stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        return pd.DataFrame({k: s for k in PRICE_COLUMNS}).sort_index()

    assert_no_lookahead(lambda c: agent.signals(as_long(c)), close, cut_points=[cut + 30])
    assert agent.report_.checkpoints and agent.report_.best_step > 0


def test_rl_agent_is_deterministic_and_validates(rl_fit, plus):
    from quantskills.strategies import RLAgent

    agent, te = rl_fit
    train = plus[plus.index.get_level_values("date") <= te]
    again = RLAgent(**RL_FAST).fit(train)
    pd.testing.assert_series_equal(agent.positions(plus), again.positions(plus))
    with pytest.raises(ValueError):
        RLAgent(state="pixels")
    with pytest.raises(KeyError):
        RLAgent(ticker="NOPE", **RL_FAST).fit(train)
