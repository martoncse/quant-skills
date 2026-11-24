from math import comb

import numpy as np
import pytest

from quantskills.robustness import pbo_cscv


def _noise(t: int = 1000, n: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, (t, n))


def test_pbo_near_half_when_every_config_is_noise():
    # Picking the best of N noise series is a coin flip out of sample.
    pbos = [pbo_cscv(_noise(seed=s), n_blocks=10).pbo for s in range(5)]
    assert 0.3 < np.mean(pbos) < 0.7


def test_pbo_near_zero_with_one_real_edge():
    m = _noise()
    m[:, 3] += 0.002  # one config has a genuine Sharpe of ~3
    r = pbo_cscv(m)
    assert r.pbo < 0.02
    assert r.prob_oos_loss < 0.02


def test_split_count_and_shapes():
    r = pbo_cscv(_noise(400, 5), n_blocks=8)
    assert r.n_splits == comb(8, 4)
    assert r.logits.shape == r.is_sharpe.shape == r.oos_sharpe.shape == (r.n_splits,)
    assert r.n_trials == 5


def test_logit_is_rank_based():
    # If OOS order always equals IS order, the IS winner is also the OOS winner.
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, (800, 1))
    m = base + np.arange(6) * 0.0005 + rng.normal(0, 1e-5, (800, 6))
    r = pbo_cscv(m, n_blocks=8)
    assert np.allclose(r.logits, np.log(6 / 1))  # rank N of N: omega = N/(N+1)
    assert r.pbo == 0.0


@pytest.mark.parametrize(
    "kwargs, msg",
    [({"n_blocks": 7}, "even"), ({"n_blocks": 0}, "even")],
)
def test_bad_blocks(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        pbo_cscv(_noise(200, 4), **kwargs)


def test_needs_two_configs_and_enough_rows():
    with pytest.raises(ValueError, match="N >= 2"):
        pbo_cscv(_noise(200, 1))
    with pytest.raises(ValueError, match="rows"):
        pbo_cscv(_noise(30, 4), n_blocks=16)
