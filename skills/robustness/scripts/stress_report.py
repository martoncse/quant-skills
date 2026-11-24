r"""Synthetic histories (#11): how realistic is each generator, and what do strategies do on them?

uv run python skills/robustness/scripts/stress_report.py --fidelity              # ~2 min
uv run python skills/robustness/scripts/stress_report.py --stress factor-long-short \
    pairs-coint vol-timing --paths 20
uv run python skills/robustness/scripts/stress_report.py --fidelity --stress \
    factor-long-short pairs-coint vol-timing --figures

--fidelity scores the three generators on a panel with real stylized facts (GARCH
volatility, Student-t tails). TimeGAN needs `uv sync --extra dl` and is skipped without it.
--stress reruns strategies on block-bootstrapped (default) histories of the **training
window** of the synthetic-plus panel, so the real hold-out is never used.
"""

import argparse
from pathlib import Path

import pandas as pd

from quantskills.data import load_cached, stylized_prices, synthetic_prices, to_wide
from quantskills.models.nn import torch_available
from quantskills.robustness import plot_generators, plot_stress
from quantskills.robustness.stress import (
    GENERATORS,
    fidelity_report,
    stress_summary,
    stress_test,
    synthetic_paths,
)
from quantskills.strategies import STRATEGIES, get_strategy
from quantskills.utils import holdout_date

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "docs" / "img" / "robustness"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fidelity", action="store_true", help="score the generators")
    p.add_argument("--stress", nargs="*", choices=list(STRATEGIES), default=[])
    p.add_argument("--method", choices=GENERATORS, default="bootstrap", help="for --stress")
    p.add_argument("--paths", type=int, default=20)
    p.add_argument("--data", choices=["stylized", "cache"], default="stylized", help="fidelity")
    p.add_argument("--timegan-steps", type=int, default=300, help="per training phase")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figures", action="store_true", help=f"save PNGs under {FIG}")
    a = p.parse_args()

    if a.fidelity:
        prices = load_cached() if a.data == "cache" else stylized_prices(seed=a.seed)
        r = to_wide(prices).pct_change(fill_method=None).iloc[1:].dropna()
        methods = [m for m in GENERATORS if m != "timegan" or torch_available()]
        steps = a.timegan_steps
        tg = {"embed_steps": steps, "supervised_steps": steps, "joint_steps": steps}
        table = fidelity_report(r, methods, n_paths=5, seed=a.seed, timegan_kwargs=tg)
        print(f"Fidelity on {a.data} data ({r.shape[1]} assets, {len(r)} days):\n")
        print(table.round(4).to_string(), "\n")
        if a.figures:
            paths = {m: synthetic_paths(r, m, 1, a.seed, **(tg if m == "timegan" else {}))[0]
                     for m in methods}  # fmt: skip
            plot_generators(r, paths, FIG / "generators.png")

    if a.stress:
        prices = synthetic_prices(seed=a.seed, n_pairs=3, garch_index=True)
        dates = prices.index.get_level_values("date")
        train = prices[dates <= holdout_date(dates, 0.3)]
        tables = {}
        for name in a.stress:
            t = stress_test(get_strategy(name), train, a.method, a.paths, a.seed)
            tables[name] = t
            s = stress_summary(t)
            print(
                f"{name:>18}: real IS Sharpe {s['is_sharpe_real']:+.2f} | {a.method} paths: "
                f"median {s['is_sharpe_median']:+.2f}, 90% range [{s['is_sharpe_p05']:+.2f}, "
                f"{s['is_sharpe_p95']:+.2f}] | real beats {s['is_sharpe_pct_real']:.0%}"
            )
        if a.figures:
            plot_stress(tables, FIG / f"stress_{a.method}.png")
        pd.set_option("display.width", 120)

    if a.figures:
        print(f"figures saved under {FIG}")


if __name__ == "__main__":
    main()
