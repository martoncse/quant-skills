"""IC and quantile report for each factor.

uv run python skills/features-alpha/scripts/factor_report.py              # synthetic data
uv run python skills/features-alpha/scripts/factor_report.py --data cache # cached prices
"""

import argparse

import pandas as pd

from quantskills.data import load_cached, synthetic_prices, to_wide
from quantskills.eval import ic_summary, information_coefficient, quantile_returns
from quantskills.features import compute_factors, forward_returns


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "cache"], default="synthetic")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prices = load_cached() if a.data == "cache" else synthetic_prices(seed=a.seed)
    close = to_wide(prices)
    if close.shape[1] < 5:
        raise SystemExit(f"Need >= 5 tickers for cross-sectional IC; got {close.shape[1]}.")
    fwd = forward_returns(close, a.horizon)
    rows, quants = {}, {}
    for name, f in compute_factors(close).items():
        rows[name] = ic_summary(information_coefficient(f, fwd), horizon=a.horizon)
        quants[name] = quantile_returns(f, fwd)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"IC vs {a.horizon}-day forward returns ({a.data} data)\n")
    print(pd.DataFrame(rows).T)
    print("\nMean forward return by quintile (1 = lowest factor)\n")
    print(pd.DataFrame(quants).T)


if __name__ == "__main__":
    main()
