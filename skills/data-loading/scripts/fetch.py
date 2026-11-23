"""Download daily prices into the parquet cache.

uv run python skills/data-loading/scripts/fetch.py --tickers SPY QQQ --start 2015-01-01
uv run python skills/data-loading/scripts/fetch.py          # example large-cap universe
"""

import argparse

from quantskills.data import EXAMPLE_LARGE_CAPS, load_prices, survivorship_warning


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cache-dir", default="data")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    tickers = a.tickers
    if tickers is None:
        survivorship_warning()
        tickers = EXAMPLE_LARGE_CAPS
    df = load_prices(tickers, a.start, a.end, cache_dir=a.cache_dir, refresh=a.refresh)
    dates = df.index.get_level_values("date")
    print(
        f"{df.index.get_level_values('ticker').nunique()} tickers, {dates.nunique()} days, "
        f"{dates.min().date()} -> {dates.max().date()}, cached in {a.cache_dir}/"
    )


if __name__ == "__main__":
    main()
