"""Matplotlib figures for the robustness write-up. Each returns the Figure."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantskills.robustness.pbo import PBOResult
from quantskills.robustness.verdict import MIN_BASE_SHARPE


def _setup(path: str | Path | None):
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, path: str | Path | None) -> None:
    if path is None:
        return
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pbo(results: Mapping[str, PBOResult], path: str | Path | None = None, title: str = "PBO"):
    """Histogram of logits per result, plus IS-vs-OOS Sharpe of the IS winner."""
    plt = _setup(path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, r in results.items():
        ax1.hist(r.logits, bins=30, alpha=0.55, label=f"{label} (PBO {r.pbo:.2f})")
        ax2.scatter(r.is_sharpe, r.oos_sharpe, s=4, alpha=0.3, label=label)
    ax1.axvline(0, color="#444", ls="--", lw=1)
    ax1.set_xlabel("logit of IS winner's OOS rank  (< 0 = below median)")
    ax1.set_ylabel("splits")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(title)
    ax2.axhline(0, color="#444", lw=0.8)
    ax2.set_xlabel("IS Sharpe of the IS winner")
    ax2.set_ylabel("its OOS Sharpe")
    ax2.set_title("Performance degradation")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_surface(
    surface: pd.DataFrame,
    path: str | Path | None = None,
    title: str = "",
    label: str = "Sharpe",
):
    """Heatmap of Sharpe over a two-parameter grid, values printed in cells."""
    plt = _setup(path)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    vals = surface.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower")
    ax.set_xticks(range(surface.shape[1]), [f"{c:g}" for c in surface.columns])
    ax.set_yticks(range(surface.shape[0]), [f"{i:g}" for i in surface.index])
    ax.set_xlabel(str(surface.columns.name))
    ax.set_ylabel(str(surface.index.name))
    for (i, j), v in np.ndenumerate(vals):
        color = "white" if np.isfinite(v) and abs(v) > 0.6 * lim else "black"
        txt = "—" if np.isnan(v) else f"{v:.2f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(title or "Parameter sensitivity")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_decay(
    curves: Mapping[str, pd.Series],
    path: str | Path | None = None,
    xlabel: str = "",
    title: str = "",
):
    """Sharpe retained (y / y[first]) against a perturbation size, one line per strategy.

    Lines whose base Sharpe is below ``MIN_BASE_SHARPE`` are left out: a ratio
    against a near-zero Sharpe is noise."""
    plt = _setup(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, s in curves.items():
        base = float(s.iloc[0])
        if base >= MIN_BASE_SHARPE:
            ax.plot(s.index, s.to_numpy() / base, marker="o", label=label)
    ax.axhline(1, color="#888", lw=0.8)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sharpe retained (1 = unchanged)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_reliability(
    tables: Mapping[str, pd.DataFrame], path: str | Path | None = None, title: str = ""
):
    """Reliability diagram: observed hit rate against mean forecast per bin."""
    plt = _setup(path)
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1, label="perfect")
    lo, hi = 1.0, 0.0
    for label, t in tables.items():
        ax.plot(t["mean_pred"], t["hit_rate"], marker="o", label=label)
        lo = min(lo, t["mean_pred"].min(), t["hit_rate"].min())
        hi = max(hi, t["mean_pred"].max(), t["hit_rate"].max())
    pad = 0.05
    ax.set_xlim(max(0, lo - pad), min(1, hi + pad))
    ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
    ax.set_xlabel("mean forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title or "Reliability diagram")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)
    return fig
