import numpy as np
import pytest

from quantskills.features import ML_FEATURES, feature_panel, from_panel, label_panel, to_panel
from quantskills.utils import assert_no_lookahead


@pytest.mark.parametrize("name", list(ML_FEATURES))
def test_ml_features_have_no_lookahead(name, close):
    assert_no_lookahead(lambda c: from_panel(feature_panel(c, [name])[name]), close)


def test_panel_round_trip(close):
    assert from_panel(to_panel(close)).equals(close)


def test_feature_panel_is_cross_sectionally_standardised(close):
    X = feature_panel(close)
    day = X.xs(close.index[-1], level="date")
    assert np.allclose(day.mean(), 0.0, atol=0.2)  # clipping can shift the mean a little
    assert set(X.columns) == set(ML_FEATURES)
    with pytest.raises(KeyError):
        feature_panel(close, ["nope"])


def test_labels_are_forward_looking_and_nan_at_the_end(close):
    h = 21
    for kind in ("raw", "rank", "up"):
        lab = from_panel(label_panel(close, h, kind))
        assert lab.iloc[-h:].isna().all().all()
        assert lab.iloc[:-h].notna().all().all()
    up = from_panel(label_panel(close, h, "up")).iloc[:-h]
    assert set(np.unique(up)) <= {0.0, 1.0}
    with pytest.raises(ValueError):
        label_panel(close, h, "bogus")
