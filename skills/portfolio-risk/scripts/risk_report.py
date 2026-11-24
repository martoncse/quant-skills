"""Weights and 1-day VaR/CVaR for a universe, over a trailing window.

uv run python skills/portfolio-risk/scripts/risk_report.py
uv run python skills/portfolio-risk/scripts/risk_report.py --method min_var --lookback 126
uv run python skills/portfolio-risk/scripts/risk_report.py --data cache --compare

Historical, in-sample risk of today's weights: a description, not a forecast.
"""

import argparse

import pandas as pd

from quantskills.data import load_cached, synthetic_prices
from quantskills.portfolio import ALLOCATORS, risk_view


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "cache"], default="synthetic")
    p.add_argument("--method", choices=ALLOCATORS, default="hrp")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--compare", action="store_true", help="table of all methods")
    a = p.parse_args()

    if a.data == "cache":
        prices = load_cached()
    else:
        prices = synthetic_prices(seed=0, n_pairs=3, garch_index=True)
    if a.compare:
        rows = {m: risk_view(prices, m, a.lookback).metrics for m in ALLOCATORS}
        print(pd.DataFrame(rows).T.round(4).to_string())
        return
    v = risk_view(prices, a.method, a.lookback)
    print(f"{v.method}: {v.lookback} days {v.start.date()} -> {v.end.date()}\n")
    print(v.table.head(15).round(4).to_string(), "\n")
    for k, val in v.metrics.items():
        print(f"{k:>14}: {val:.4f}")


if __name__ == "__main__":
    main()
