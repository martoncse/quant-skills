"""Tear sheet: headline metrics + cost/turnover + deflated Sharpe, and a plot."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantskills.backtest.engine import BacktestResult
from quantskills.eval.metrics import TRADING_DAYS, drawdown_series, summary
from quantskills.eval.sharpe import deflated_sharpe_ratio


def tear_sheet(
    result: BacktestResult,
    n_trials: int = 1,
    trial_sharpes: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarise a backtest. ``n_trials`` must count *every* configuration tried.

    The leading warm-up period with no positions (e.g. while a 12-month
    lookback fills) is dropped so it does not dilute the statistics.
    """
    result = result.trim_warmup()
    out: dict[str, Any] = summary(result.returns)
    out["gross_sharpe"] = summary(result.gross_returns)["sharpe"]
    out["ann_turnover"] = float(result.turnover.mean() * TRADING_DAYS)
    out["ann_cost_drag"] = float(result.costs.mean() * TRADING_DAYS)
    out["cost_bps"] = result.cost_model.rate * 1e4
    dsr = deflated_sharpe_ratio(result.returns, n_trials=n_trials, trial_sharpes=trial_sharpes)
    out["psr"] = dsr.psr
    out["dsr"] = dsr.dsr
    out["sharpe_threshold"] = dsr.sharpe_threshold
    out["n_trials"] = dsr.n_trials
    out["start"] = str(result.returns.index[0].date())
    out["end"] = str(result.returns.index[-1].date())
    return out


_FMT = {
    "ann_return": "{:.2%}",
    "ann_vol": "{:.2%}",
    "max_drawdown": "{:.2%}",
    "hit_rate": "{:.1%}",
    "ann_turnover": "{:.1f}x",
    "ann_cost_drag": "{:.2%}",
    "psr": "{:.3f}",
    "dsr": "{:.3f}",
    "n_days": "{:.0f}",
    "cost_bps": "{:.1f}",
}


def format_tear_sheet(sheet: dict[str, Any], title: str = "Tear sheet") -> str:
    lines = [title, "-" * len(title)]
    for k, v in sheet.items():
        if isinstance(v, float):
            v = _FMT.get(k, "{:.3f}").format(v)
        lines.append(f"{k:<18}{v}")
    lines.append("Backtested on historical/synthetic data; not a forecast of real returns.")
    return "\n".join(lines)


def plot_tear_sheet(
    result: BacktestResult,
    path: str | Path | None = None,
    title: str = "Backtest",
    split_date: str | pd.Timestamp | None = None,
):
    """Equity (net vs gross) and drawdown. Returns the matplotlib Figure."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    idx = result.returns.index
    gross_eq = (1 + result.gross_returns).cumprod().to_numpy()
    ax1.plot(idx, gross_eq, label="gross", color="#9aa5b1", lw=1)
    ax1.plot(idx, result.equity.to_numpy(), label="net of costs", color="#1f6feb", lw=1.5)
    if split_date is not None:
        for ax in (ax1, ax2):
            ax.axvline(pd.Timestamp(split_date), color="#444", ls="--", lw=1)
        ax1.text(pd.Timestamp(split_date), ax1.get_ylim()[1], "  hold-out →", va="top")
    ax1.set_title(title)
    ax1.set_ylabel("growth of $1")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)
    dd = drawdown_series(result.returns)
    ax2.fill_between(dd.index, dd.to_numpy(), 0, color="#d1242f", alpha=0.4)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig
