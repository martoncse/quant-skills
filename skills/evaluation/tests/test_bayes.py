import numpy as np
import pandas as pd
import pytest

from quantskills.eval import bayesian_sharpe, sharpe_ratio


def test_posterior_centres_on_sample_sharpe_and_is_reproducible():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    post = bayesian_sharpe(r, seed=1)
    assert post.mean == pytest.approx(sharpe_ratio(r), abs=0.05)
    assert post.ci_low < sharpe_ratio(r) < post.ci_high
    assert bayesian_sharpe(r, seed=1).mean == post.mean


def test_more_data_narrows_the_posterior():
    rng = np.random.default_rng(2)
    short = bayesian_sharpe(pd.Series(rng.normal(0.0005, 0.01, 250)))
    long = bayesian_sharpe(pd.Series(rng.normal(0.0005, 0.01, 2500)))
    assert (long.ci_high - long.ci_low) < (short.ci_high - short.ci_low) / 2


def test_prob_positive_is_sensible():
    rng = np.random.default_rng(3)
    good = bayesian_sharpe(pd.Series(rng.normal(0.002, 0.01, 1000)))
    bad = bayesian_sharpe(pd.Series(rng.normal(-0.002, 0.01, 1000)))
    assert good.prob_positive > 0.99 and bad.prob_positive < 0.01
    with pytest.raises(ValueError):
        bayesian_sharpe(pd.Series([0.01, 0.02]))
