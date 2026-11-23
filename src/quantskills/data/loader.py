"""Price loading with a parquet cache.

This is the only module in ``src/`` allowed to touch the network (via yfinance).
All outputs follow the repo convention: long format, ``MultiIndex(date, ticker)``,
tz-naive dates sorted ascending, columns ``open, high, low, close, volume``.
Prices are split/dividend adjusted (``auto_adjust=True``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
INDEX_NAMES = ["date", "ticker"]


def _download_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download one ticker's daily adjusted OHLCV from Yahoo Finance."""
    import yfinance as yf  # imported lazily so tests never need the network

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    return raw[PRICE_COLUMNS]


def _normalise_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["ticker"] = ticker
    return out.set_index("ticker", append=True)[PRICE_COLUMNS]


def _cache_file(cache_dir: Path, ticker: str, start: str, end: str | None) -> Path:
    safe = ticker.replace("/", "_").replace("^", "IDX_")
    return cache_dir / f"{safe}__{start}__{end or 'latest'}.parquet"


def load_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data",
    refresh: bool = False,
    downloader: Callable[[str, str, str | None], pd.DataFrame] = _download_yfinance,
) -> pd.DataFrame:
    """Load daily OHLCV for ``tickers``, using the parquet cache when present.

    Each (ticker, start, end) request is cached separately under ``cache_dir``.
    ``downloader`` can be swapped (e.g. in tests) for any function returning a
    date-indexed OHLCV frame.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for ticker in tickers:
        path = _cache_file(cache, ticker, start, end)
        if path.exists() and not refresh:
            frames.append(pd.read_parquet(path))
            continue
        df = _normalise_single(downloader(ticker, start, end), ticker)
        df.to_parquet(path)
        frames.append(df)
    if not frames:
        raise ValueError("No tickers given")
    out = pd.concat(frames).sort_index()
    validate_prices(out)
    return out


def load_cached(cache_dir: str | Path = "data") -> pd.DataFrame:
    """Load everything in the cache (offline). Raises ``FileNotFoundError`` if empty."""
    files = sorted(Path(cache_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Price cache {cache_dir!s} is empty. Run "
            "`uv run python skills/data-loading/scripts/fetch.py` first."
        )
    out = pd.concat(pd.read_parquet(f) for f in files)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    validate_prices(out)
    return out


def validate_prices(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` breaks the long-format price conventions."""
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != INDEX_NAMES:
        raise ValueError(f"Index must be MultiIndex{tuple(INDEX_NAMES)}")
    dates = df.index.get_level_values("date")
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("date level must be a DatetimeIndex")
    if dates.tz is not None:
        raise ValueError("dates must be tz-naive (UTC dates)")
    if df.index.has_duplicates:
        raise ValueError("duplicate (date, ticker) rows")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if (df["close"].dropna() <= 0).any():
        raise ValueError("close prices must be positive")


def to_wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Long ``(date, ticker)`` frame -> wide ``date x ticker`` frame for ``field``."""
    wide = df[field].unstack("ticker").sort_index()
    wide.columns.name = "ticker"
    return wide


def to_long(wide: pd.DataFrame, name: str) -> pd.Series:
    """Wide ``date x ticker`` frame -> long Series indexed ``(date, ticker)``."""
    s = wide.stack(future_stack=True).dropna()
    s.index.names = INDEX_NAMES
    s.name = name
    return s.sort_index()
