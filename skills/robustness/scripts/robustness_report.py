r"""Robustness report: DSR, PBO, noise / delay / sensitivity, regimes, graded R/A/G.

uv run python skills/robustness/scripts/robustness_report.py                    # strategy #1
uv run python skills/robustness/scripts/robustness_report.py --strategy pairs-coint \
    --param entry_z=2.5
uv run python skills/robustness/scripts/robustness_report.py --all              # ~3 min
uv run python skills/robustness/scripts/robustness_report.py --all --figures    # + docs/img

Every check runs walk-forward inside the training window. The hold-out appears
once, in the last row. Re-running is fine; re-tuning after reading that row is not.
"""

import argparse
import ast
from pathlib import Path

import pandas as pd

from quantskills.data import load_cached, synthetic_prices
from quantskills.robustness import (
    RobustnessReport,
    plot_decay,
    plot_pbo,
    plot_surface,
    robustness_report,
)
from quantskills.strategies import STRATEGIES, get_strategy
from quantskills.utils import holdout_date

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "docs" / "img" / "robustness"


def _parse(kv: str) -> tuple[str, object]:
    k, v = kv.split("=", 1)
    try:
        return k, ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return k, v


def _value(r: RobustnessReport, name: str) -> float:
    return next((c.value for c in r.checks if c.name.startswith(name)), float("nan"))


def _summary_row(i: int, r: RobustnessReport) -> str:
    def f(x: float) -> str:
        return "—" if pd.isna(x) else f"{x:.2f}"

    p_loss = r.pbo.prob_oos_loss if r.pbo else float("nan")
    return (
        f"| {i} | `{r.strategy}` | **{r.verdict.upper()}** | {r.dsr.dsr:.3f} "
        f"| {f(_value(r, 'PBO'))} / {f(p_loss)} | {f(_value(r, 'Price noise'))} "
        f"| {f(_value(r, 'Trade 1 day'))} | {f(r.is_sharpe)} → {f(r.wf_sharpe)} "
        f"| {f(r.oos_sharpe)} |"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--strategy", choices=list(STRATEGIES), default="factor-long-short")
    p.add_argument("--all", action="store_true", help="every strategy, one summary table")
    p.add_argument(
        "--data", choices=["synthetic-plus", "synthetic", "cache"], default="synthetic-plus"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["walk-forward", "in-sample"], default="walk-forward")
    p.add_argument("--n-trials", type=int, default=1, help="configs tried before this report")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--figures", action="store_true", help=f"save PNGs under {FIG}")
    a = p.parse_args()

    if a.data == "cache":
        prices = load_cached()
    elif a.data == "synthetic-plus":
        prices = synthetic_prices(seed=a.seed, n_pairs=3, garch_index=True)
    else:
        prices = synthetic_prices(seed=a.seed)
    train_end = holdout_date(prices.index.get_level_values("date"), 0.3)
    names = list(STRATEGIES) if a.all else [a.strategy]
    params = dict(_parse(kv) for kv in a.param)

    reports: dict[str, RobustnessReport] = {}
    for name in names:
        strategy = get_strategy(name, **(params if not a.all else {}))
        r = robustness_report(
            strategy,
            prices,
            train_end,
            mode=a.mode,
            n_trials=a.n_trials,
            progress=lambda m, n=name: print(f"  [{n}] {m}", flush=True),
        )
        reports[name] = r
        print()
        print(r.to_markdown())
        if r.surface is not None:
            print(f"\nSensitivity surface ({r.mode} Sharpe):\n{r.surface.round(2).to_string()}")
        print(f"\nNoise:\n{r.noise_summary().round(2).T.to_string()}")
        print(f"\nDelay:\n{r.delay.set_index('delay').round(2).T.to_string()}")
        print(f"\nRegimes:\n{r.regimes.round(2).to_string()}\n")

    if a.all:
        print(
            "| # | Strategy | Verdict | DSR | PBO / P(loss) | Noise kept | 1-day delay kept "
            "| Sharpe IS → walk-fwd | Hold-out Sharpe |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(reports.values(), start=1):
            print(_summary_row(i, r))

    if a.figures:
        FIG.mkdir(parents=True, exist_ok=True)
        pbos = {n: r.pbo for n, r in reports.items() if r.pbo is not None}
        if pbos:
            pick = {n: pbos[n] for n in ("pairs-coint", "gbm-long-short") if n in pbos} or pbos
            plot_pbo(pick, FIG / "pbo.png", title="CSCV: where does the IS winner land OOS?")
        for n, r in reports.items():
            if r.surface is not None:
                plot_surface(r.surface, FIG / f"surface_{n}.png", title=n, label=f"{r.mode} Sharpe")
        noise = {n: r.noise_summary()["sharpe"] for n, r in reports.items()}
        plot_decay(noise, FIG / "noise.png", "price noise (bps)", "Refit on noisy prices")
        delay = {n: r.delay.set_index("delay")["sharpe"] for n, r in reports.items()}
        plot_decay(delay, FIG / "delay.png", "execution delay (days)", "Trade k days late")
        print(f"figures saved under {FIG}")


if __name__ == "__main__":
    main()
