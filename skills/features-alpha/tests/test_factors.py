import numpy as np
import pandas as pd
import pytest

from quantskills.features import (
    FACTORS,
    compute_factors,
    forward_returns,
    momentum,
    rank_cross_section,
    zscore_cross_section,
)
from quantskills.utils import assert_no_lookahead


@pytest.mark.parametrize("name", sorted(FACTORS))
def test_factor_has_no_lookahead(name, close):
    assert_no_lookahead(FACTORS[name], close)


def test_zscore_and_rank_have_no_lookahead(close):
    assert_no_lookahead(lambda c: zscore_cross_section(momentum(c)), close)
    assert_no_lookahead(lambda c: rank_cross_section(momentum(c)), close)


def test_lookahead_checker_catches_a_leak(close):
    with pytest.raises(AssertionError):
        assert_no_lookahead(lambda c: forward_returns(c, 5), close)


def test_forward_returns_is_shifted_label(close):
    fwd = forward_returns(close, 3)
    t = 100
    expected = close.iloc[t + 3] / close.iloc[t] - 1
    pd.testing.assert_series_equal(fwd.iloc[t], expected, check_names=False)
    assert fwd.iloc[-3:].isna().all().all()


def test_momentum_formula():
    close = pd.DataFrame({"A": np.arange(1.0, 301.0)})
    m = momentum(close, lookback=252, skip=21)
    t = 280
    assert m.iloc[t, 0] == pytest.approx(close.iloc[t - 21, 0] / close.iloc[t - 252, 0] - 1)
    assert m.iloc[:252].isna().all().all()


def test_zscore_is_standardised(close):
    z = zscore_cross_section(momentum(close), clip=100).dropna(how="all")
    assert np.allclose(z.mean(axis=1), 0, atol=1e-12)
    assert np.allclose(z.std(axis=1), 1, atol=1e-12)


def test_compute_factors_rejects_unknown(close):
    with pytest.raises(KeyError):
        compute_factors(close, ["value"])
