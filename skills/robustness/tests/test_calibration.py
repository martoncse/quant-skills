import numpy as np
import pytest

from quantskills.robustness import (
    brier_decomposition,
    brier_score,
    calibration_summary,
    expected_calibration_error,
    log_loss,
    reliability_table,
)


def _calibrated(n: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(float)  # y ~ Bernoulli(p): calibrated by construction
    return p, y


def test_constant_half_scores_quarter():
    y = np.array([0, 1, 1, 0, 1.0])
    assert brier_score(np.full(5, 0.5), y) == pytest.approx(0.25)
    assert log_loss(np.full(5, 0.5), y) == pytest.approx(np.log(2))


def test_calibrated_forecasts_have_small_ece():
    p, y = _calibrated()
    assert expected_calibration_error(p, y) < 0.02
    t = reliability_table(p, y, 10)
    assert (t["gap"].abs() < 0.04).all()
    assert t["count"].sum() == len(p)


def test_overconfidence_raises_ece_and_brier():
    p, y = _calibrated()
    logit = np.log(p / (1 - p))
    sharp = 1 / (1 + np.exp(-3 * logit))  # same ranking, pushed towards 0 and 1
    assert expected_calibration_error(sharp, y) > 3 * expected_calibration_error(p, y)
    assert brier_score(sharp, y) > brier_score(p, y)


def test_decomposition_adds_up_for_binned_forecasts():
    rng = np.random.default_rng(2)
    p = rng.choice([0.2, 0.4, 0.6, 0.8], 5000)  # constant within bins: identity is exact
    y = (rng.uniform(size=5000) < np.clip(p + 0.1, 0, 1)).astype(float)
    d = brier_decomposition(p, y, n_bins=4, strategy="uniform")
    assert d["reliability"] - d["resolution"] + d["uncertainty"] == pytest.approx(d["brier"])


def test_summary_keys_and_skill_sign():
    p, y = _calibrated()
    s = calibration_summary(p, y)
    assert {"brier", "ece", "log_loss", "brier_skill", "reliability", "resolution"} <= set(s)
    assert s["brier_skill"] > 0  # informative and calibrated beats the base rate


def test_identical_forecasts_make_one_bin():
    t = reliability_table(np.full(10, 0.3), np.r_[np.ones(3), np.zeros(7)])
    assert len(t) == 1 and t["hit_rate"].iloc[0] == pytest.approx(0.3)


def test_bad_inputs():
    with pytest.raises(ValueError, match="same shape"):
        brier_score([0.1, 0.2], [1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1.2], [1])
    with pytest.raises(ValueError, match="0 or 1"):
        brier_score([0.5], [2])
    with pytest.raises(ValueError, match="strategy"):
        reliability_table([0.5], [1], strategy="magic")
    # NaNs are dropped, not propagated
    assert brier_score([0.5, np.nan], [1, 0]) == pytest.approx(0.25)
