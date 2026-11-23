import pandas as pd
import pytest

from quantskills.data import (
    PRICE_COLUMNS,
    load_cached,
    load_prices,
    survivorship_warning,
    synthetic_prices,
    to_long,
    to_wide,
    validate_prices,
)
from quantskills.data.universe import SurvivorshipBiasWarning


def _fake_download(ticker, start, end):
    idx = pd.date_range(start, periods=5, freq="B", tz="America/New_York")
    base = 10.0 + len(ticker)
    return pd.DataFrame({c: [base + i for i in range(5)] for c in PRICE_COLUMNS}, index=idx)


def test_synthetic_follows_conventions():
    df = synthetic_prices(n_assets=5, n_days=50, seed=1)
    validate_prices(df)
    assert df.shape == (250, 5)
    assert list(df.columns) == PRICE_COLUMNS


def test_synthetic_is_deterministic_with_seed():
    a = synthetic_prices(n_assets=3, n_days=20, seed=3)
    b = synthetic_prices(n_assets=3, n_days=20, seed=3)
    c = synthetic_prices(n_assets=3, n_days=20, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_load_prices_caches_and_normalises(tmp_path):
    calls = []

    def dl(t, s, e):
        calls.append(t)
        return _fake_download(t, s, e)

    df = load_prices(["AAA", "BB"], "2024-01-01", cache_dir=tmp_path, downloader=dl)
    validate_prices(df)
    assert df.index.get_level_values("date").tz is None
    assert set(df.index.get_level_values("ticker")) == {"AAA", "BB"}
    assert calls == ["AAA", "BB"]

    again = load_prices(["AAA", "BB"], "2024-01-01", cache_dir=tmp_path, downloader=dl)
    assert calls == ["AAA", "BB"], "second call should be served from cache"
    pd.testing.assert_frame_equal(df, again)

    pd.testing.assert_frame_equal(load_cached(tmp_path), df)


def test_load_cached_empty_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="empty"):
        load_cached(tmp_path)


def test_validate_rejects_bad_frames():
    df = synthetic_prices(n_assets=2, n_days=10, seed=0)
    with pytest.raises(ValueError, match="sorted"):
        validate_prices(df.iloc[::-1])
    with pytest.raises(ValueError, match="missing"):
        validate_prices(df.drop(columns="volume"))
    with pytest.raises(ValueError, match="MultiIndex"):
        validate_prices(df.reset_index(level="ticker"))


def test_wide_long_roundtrip():
    df = synthetic_prices(n_assets=3, n_days=10, seed=0)
    wide = to_wide(df)
    assert wide.shape == (10, 3)
    long = to_long(wide, "close")
    pd.testing.assert_series_equal(long, df["close"])


def test_survivorship_warning():
    with pytest.warns(SurvivorshipBiasWarning):
        survivorship_warning()
