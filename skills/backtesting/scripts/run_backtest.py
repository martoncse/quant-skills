"""Fit a strategy on the training window and backtest it with costs.

uv run python skills/backtesting/scripts/run_backtest.py
uv run python skills/backtesting/scripts/run_backtest.py --data cache --plot out.png
uv run python skills/backtesting/scripts/run_backtest.py --param quantile=0.1
"""

import argparse
import ast

from quantskills.backtest import CostModel
from quantskills.data import load_cached, synthetic_prices
from quantskills.eval import format_tear_sheet, plot_tear_sheet, tear_sheet
from quantskills.strategies import STRATEGIES, get_strategy, run_strategy
from quantskills.utils import holdout_date


def _parse(kv: str) -> tuple[str, object]:
    k, v = kv.split("=", 1)
    try:
        return k, ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return k, v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--strategy", choices=sorted(STRATEGIES), default="factor-long-short")
    p.add_argument("--data", choices=["synthetic", "cache"], default="synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--commission-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--n-trials", type=int, default=1, help="configs tried so far (for DSR)")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--plot", default=None, help="save an equity/drawdown PNG here")
    a = p.parse_args()

    prices = load_cached() if a.data == "cache" else synthetic_prices(seed=a.seed)
    train_end = holdout_date(prices.index.get_level_values("date"), a.test_fraction)
    strat = get_strategy(a.strategy, **dict(_parse(kv) for kv in a.param))
    run = run_strategy(strat, prices, train_end, CostModel(a.commission_bps, a.slippage_bps))

    print(strat)
    print(f"train <= {train_end.date()}\n")
    for label, res in (("In-sample", run.in_sample), ("Out-of-sample", run.out_of_sample)):
        print(format_tear_sheet(tear_sheet(res, n_trials=a.n_trials), title=label), "\n")
    if a.plot:
        title = f"{a.strategy} ({a.data} data, net of {a.commission_bps + a.slippage_bps:g} bps)"
        plot_tear_sheet(run.full, a.plot, title=title, split_date=train_end)
        print(f"saved {a.plot}")


if __name__ == "__main__":
    main()
