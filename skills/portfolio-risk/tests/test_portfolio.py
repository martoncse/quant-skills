import numpy as np
import pandas as pd
import pytest

from quantskills.features import simple_returns
from quantskills.portfolio import (
    ALLOCATORS,
    allocate,
    cluster_order,
    cornish_fisher_var,
    cov_to_corr,
    eigenportfolio,
    historical_cvar,
    historical_var,
    hrp_weights,
    inverse_volatility,
    ledoit_wolf_cov,
    max_sharpe,
    mean_variance,
    min_variance,
    parametric_var,
    pca_factors,
    risk_contributions,
    risk_report,
    sample_cov,
)


@pytest.fixture(scope="module")
def rets(close):
    return simple_returns(close).iloc[-500:]


@pytest.fixture(scope="module")
def cov(rets):
    return ledoit_wolf_cov(rets)


def test_covariances_are_symmetric_psd(rets):
    for c in (sample_cov(rets), ledoit_wolf_cov(rets)):
        np.testing.assert_allclose(c, c.T)
        assert np.linalg.eigvalsh(c).min() > -1e-12
    corr = cov_to_corr(ledoit_wolf_cov(rets))
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_min_variance_beats_equal_weight_in_sample(cov):
    w = min_variance(cov)
    eq = np.full(len(cov), 1 / len(cov))
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    assert w @ cov @ w <= eq @ cov.to_numpy() @ eq + 1e-12
    capped = min_variance(cov, max_weight=0.1)
    assert capped.max() <= 0.1 + 1e-9
    with pytest.raises(ValueError):
        min_variance(cov, max_weight=0.01)


def test_min_variance_matches_closed_form_when_unconstrained():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=list("ab"), columns=list("ab"))
    inv = np.linalg.inv(cov.to_numpy())
    exact = inv.sum(axis=1) / inv.sum()
    np.testing.assert_allclose(min_variance(cov, long_only=False), exact, atol=1e-6)


def test_mean_variance_and_max_sharpe_tilt_to_higher_return(cov):
    mu = pd.Series(0.05, index=cov.index)
    mu.iloc[0] = 0.5
    for w in (mean_variance(mu, cov, risk_aversion=2.0), max_sharpe(mu, cov)):
        assert w.sum() == pytest.approx(1.0)
        assert w.iloc[0] == w.max()


def test_hrp_is_long_only_and_groups_correlated_assets():
    rng = np.random.default_rng(0)
    f1, f2 = rng.normal(0, 0.01, (2, 600))
    data = {f"A{i}": f1 + rng.normal(0, 0.003, 600) for i in range(3)}
    data |= {f"B{i}": f2 + rng.normal(0, 0.003, 600) for i in range(3)}
    r = pd.DataFrame(data)
    order = cluster_order(r.corr())
    groups = ["A" if t.startswith("A") else "B" for t in order]
    assert groups in (["A"] * 3 + ["B"] * 3, ["B"] * 3 + ["A"] * 3)
    w = hrp_weights(sample_cov(r))
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    # two equal-risk clusters get roughly half the budget each
    assert w[[c for c in w.index if c.startswith("A")]].sum() == pytest.approx(0.5, abs=0.1)


def test_hrp_on_diagonal_cov_is_inverse_variance():
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.09, 0.16]), index=list("abcd"), columns=list("abcd"))
    ivp = (1 / np.diag(cov)) / (1 / np.diag(cov)).sum()
    np.testing.assert_allclose(hrp_weights(cov), ivp, atol=1e-12)
    np.testing.assert_allclose(inverse_volatility(cov).sum(), 1.0)


def test_pca_first_component_is_the_market(rets):
    res = pca_factors(rets)
    assert res.explained.iloc[0] == res.explained.max()
    assert res.explained.sum() == pytest.approx(1.0)
    assert (res.loadings["PC1"] > 0).all()  # one-factor synthetic data
    w = eigenportfolio(rets, k=1)
    assert w.abs().sum() == pytest.approx(1.0) and (w > 0).any() and (w < 0).any()


def test_var_cvar_on_known_distribution():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200_000))
    assert historical_var(r, 0.95) == pytest.approx(0.01645, rel=0.02)
    assert parametric_var(r, 0.99) == pytest.approx(0.02326, rel=0.02)
    assert cornish_fisher_var(r, 0.99) == pytest.approx(0.02326, rel=0.03)
    assert historical_cvar(r, 0.95) == pytest.approx(0.02063, rel=0.02)  # phi(z)/(1-a)
    assert historical_cvar(r, 0.95) > historical_var(r, 0.95)
    with pytest.raises(ValueError):
        historical_var(r, 5)


def test_cornish_fisher_widens_with_fat_tails():
    r = pd.Series(np.random.default_rng(2).standard_t(3, 50_000) * 0.01)
    assert cornish_fisher_var(r, 0.99) > parametric_var(r, 0.99)


def test_risk_contributions_sum_to_one(cov):
    w = hrp_weights(cov)
    rc = risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("method", ALLOCATORS)
def test_allocate_and_report(rets, method):
    r = rets.copy()
    r.iloc[:10, 0] = np.nan  # incomplete column is dropped, gets weight 0
    w = allocate(r, method)
    assert w.iloc[0] == 0.0
    total = w.abs().sum() if method == "eigen" else w.sum()
    assert total == pytest.approx(1.0)
    rep = risk_report(rets, w)
    assert rep["cvar_hist_99"] >= rep["var_hist_99"] > 0
    with pytest.raises(ValueError):
        allocate(rets, "bogus")
