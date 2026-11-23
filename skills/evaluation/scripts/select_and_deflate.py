"""Honest model selection: sweep params in-sample, deflate the winner's Sharpe,
then look at the hold-out exactly once.

    uv run python skills/evaluation/scripts/select_and_deflate.py
"""

import argparse
import itertools

from quantskills.data import load_cached, synthetic_prices
from quantskills.eval import deflated_sharpe_ratio, format_tear_sheet, tear_sheet
from quantskills.strategies import FactorLongShort, run_strategy, sweep_params
from quantskills.utils import holdout_date


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "cache"], default="synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-fraction", type=float, default=0.3)
    a = p.parse_args()

    prices = load_cached() if a.data == "cache" else synthetic_prices(seed=a.seed)
    train_end = holdout_date(prices.index.get_level_values("date"), a.test_fraction)
    grid = [
        {"quantile": q, "rebalance_days": r, "weighting": w}
        for q, r, w in itertools.product((0.1, 0.2, 0.3), (5, 21), ("ic", "equal"))
    ]
    table = sweep_params(FactorLongShort, grid, prices, train_end)
    print(table.sort_values("is_sharpe", ascending=False).to_string(index=False), "\n")

    best = table.loc[table["is_sharpe"].idxmax()].drop("is_sharpe").to_dict()
    best["rebalance_days"] = int(best["rebalance_days"])
    run = run_strategy(FactorLongShort(**best), prices, train_end)
    trials = table["is_sharpe"].tolist()
    d = deflated_sharpe_ratio(run.in_sample.trim_warmup().returns, trial_sharpes=trials)
    print(
        f"best of {d.n_trials}: IS Sharpe {d.sharpe:.2f}, null threshold "
        f"{d.sharpe_threshold:.2f}, PSR {d.psr:.3f}, DSR {d.dsr:.3f}\n"
    )
    sheet = tear_sheet(run.out_of_sample, trial_sharpes=trials)
    print(format_tear_sheet(sheet, title="Hold-out (touched once)"))


if __name__ == "__main__":
    main()
