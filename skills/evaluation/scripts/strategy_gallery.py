"""Run every registered strategy once, with default params, on the same synthetic panel.

uv run python skills/evaluation/scripts/strategy_gallery.py
uv run python skills/evaluation/scripts/strategy_gallery.py --figures   # also save PNGs

Panel: 30 one-factor stocks + 3 cointegrated pairs + a GARCH index (seed 0) with
synthetic headlines attached (strategy #8),
last 30% held out, 5 + 5 bps costs. Prints a markdown table for the README.

Honesty note: N=1 per strategy here, but the defaults were chosen while
developing against synthetic data, so treat in-sample numbers as tuned.
Results are illustrations of the pipeline, not evidence about markets.
"""

import argparse
from pathlib import Path

from quantskills.data import synthetic_prices_with_news
from quantskills.eval import bayesian_sharpe, plot_tear_sheet, tear_sheet
from quantskills.strategies import STRATEGIES, run_strategy
from quantskills.utils import holdout_date

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figures", action="store_true", help="save equity PNGs under strategies/")
    a = p.parse_args()

    prices = synthetic_prices_with_news(seed=a.seed)  # same prices + headlines for #8
    train_end = holdout_date(prices.index.get_level_values("date"), 0.3)
    print(f"train <= {train_end.date()}, costs 10 bps\n")
    print(
        "| # | Strategy | IS Sharpe | Hold-out Sharpe | Hold-out 90% CI (Bayes) "
        "| Hold-out max DD | Turnover/yr | Hold-out DSR (N=1) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for i, (name, cls) in enumerate(STRATEGIES.items(), start=1):
        run = run_strategy(cls(), prices, train_end)
        is_, oos = tear_sheet(run.in_sample), tear_sheet(run.out_of_sample)
        b = bayesian_sharpe(run.out_of_sample.trim_warmup().returns, seed=a.seed)
        print(
            f"| {i} | `{name}` | {is_['sharpe']:.2f} | **{oos['sharpe']:.2f}** "
            f"| [{b.ci_low:.2f}, {b.ci_high:.2f}] | {oos['max_drawdown']:.1%} "
            f"| {oos['ann_turnover']:.1f}x | {oos['dsr']:.2f} |"
        )
        if a.figures:
            out = ROOT / "strategies" / name / "figures" / "synthetic_plus_equity.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plot_tear_sheet(
                run.full, out, title=f"{name} (synthetic, net of 10 bps)", split_date=train_end
            )


if __name__ == "__main__":
    main()
