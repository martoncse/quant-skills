---
name: data-loading
description: Load, cache and validate daily OHLCV price panels (Yahoo Finance or synthetic) in the repo's long MultiIndex(date, ticker) format.
---

## When to use
Any time you need prices: before computing features, backtesting a strategy,
or populating the terminal. Use the synthetic generator for tests, demos, and
for checking that a method recovers a planted signal and finds nothing in noise.

## Inputs / outputs
- `load_prices(tickers, start, end=None, cache_dir="data", refresh=False) -> DataFrame`
  - Network only on a cache miss. One parquet file per (ticker, start, end).
- `load_cached(cache_dir="data") -> DataFrame`: offline; raises a clear
  `FileNotFoundError` when the cache is empty.
- `synthetic_prices(n_assets, n_days, seed, drift_vol=...) -> DataFrame`:
  one-factor model plus persistent per-asset drift (a planted momentum effect).
  `drift_vol=0` gives a no-signal null panel.
- Output convention (checked by `validate_prices`):
  - index `MultiIndex(date, ticker)`, dates tz-naive, sorted ascending, no duplicates
  - columns `open, high, low, close, volume`; prices split/dividend adjusted
- `to_wide(df, "close")` gives a `date x ticker` frame; `to_long(wide, name)` is the inverse.

## Steps
1. `uv run python skills/data-loading/scripts/fetch.py --tickers SPY QQQ IWM --start 2015-01-01`
2. In code: `prices = load_cached()` (offline) or `load_prices([...], "2015-01-01")`.
3. `close = to_wide(prices)` for factor/backtest code.

## Pitfalls
- **Survivorship bias.** `EXAMPLE_LARGE_CAPS` is today's large caps.
  `fetch.py` emits a `SurvivorshipBiasWarning` when you use it. Results on it
  are upward biased because delisted and shrunken names are missing.
- Yahoo adjusted prices are revised retroactively after new dividends and
  splits. Re-downloading can change history, so pin cache files for reproducible runs.
- Missing days per ticker show up as NaN in wide frames. Don't forward-fill
  returns. The backtester treats a NaN return as 0 for a held position.
- Fundamentals from yfinance are *not* point-in-time. Don't use them as features.
- Never call the network from tests: inject `downloader=` or use `synthetic_prices`.

## References
- ML4T 3rd ed., ch. 2 (financial data universe, survivorship bias)
- yfinance (Apache-2.0): https://github.com/ranaroussi/yfinance
