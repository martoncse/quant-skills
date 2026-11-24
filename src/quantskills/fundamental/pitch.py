"""One-page equity pitch: valuation summary, football field, sensitivity, and the analyst's case.

The generator does the arithmetic and the layout. The **thesis, risks and catalysts
come from the case file**, written by a person: nothing here invents a narrative, and
empty sections say so. The label at the top is mechanical (base DCF vs price), not a
recommendation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quantskills.fundamental.case import Case
from quantskills.fundamental.comps import comps_range, year1_metrics
from quantskills.fundamental.dcf import (
    cost_of_equity,
    dcf_value,
    implied_growth,
    project,
    scenario_table,
    sensitivity,
)

DISCLAIMER = (
    "Illustrative analysis of stated assumptions. Not investment advice: a DCF is only as "
    "good as its inputs, and small changes in WACC or terminal growth move the value a lot."
)


def valuation_label(upside: float, band: float = 0.15) -> str:
    """Mechanical read of base DCF vs price; not a buy / sell call."""
    if upside > band:
        return f"Base-case value is {upside:.0%} above the price"
    if upside < -band:
        return f"Base-case value is {-upside:.0%} below the price"
    return "Base-case value is within 15% of the price"


def football_field(case: Case) -> pd.DataFrame:
    """Value-per-share ranges by method (low, mid, high), for the chart and the summary."""
    rows: dict[str, dict[str, float]] = {}
    sens = sensitivity(case, (-0.005, 0.0, 0.005), (-0.005, 0.0, 0.005))
    base = dcf_value(case).per_share
    rows["DCF (WACC ±0.5pp, g ±0.5pp)"] = {
        "low": float(sens.min().min()),
        "mid": base,
        "high": float(sens.max().max()),
    }
    if case.scenarios:
        st = scenario_table(case)["per_share"]
        rows["DCF scenarios"] = {"low": st.min(), "mid": st["base"], "high": st.max()}
    if case.terminal.exit_ev_ebitda is not None:
        v = dcf_value(case, method="exit").per_share
        rows[f"DCF, exit {case.terminal.exit_ev_ebitda:g}x EBITDA"] = {
            "low": v,
            "mid": v,
            "high": v,
        }
    for label, r in comps_range(case).iterrows():
        rows[str(label)] = {"low": r["low"], "mid": r["mid"], "high": r["high"]}
    return pd.DataFrame(rows).T


def plot_football_field(case: Case, path: str | Path | None = None):
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ff = football_field(case).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(ff) + 1.4))
    y = range(len(ff))
    ax.barh(y, ff["high"] - ff["low"], left=ff["low"], color="#9ecae1", height=0.5)
    ax.scatter(ff["mid"], y, color="#08519c", zorder=3, s=18, label="central value")
    ax.axvline(case.company.price, color="#d1242f", lw=1.5, label=f"price {case.company.price:g}")
    ax.set_yticks(list(y), list(ff.index))
    lo = min(ff["low"].min(), case.company.price)
    hi = max(ff["high"].max(), case.company.price)
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(f"value per share ({case.company.currency})")
    title = f"{case.company.name} ({case.company.ticker}): valuation range"
    ax.set_title(title + (" [FICTIONAL]" if case.company.fictional else ""))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig


def _bullets(items: list[str], empty: str) -> str:
    return "\n".join(f"- {x}" for x in items) if items else f"- _{empty}_"


def _md_table(df: pd.DataFrame, fmt: str = "{:,.2f}") -> str:
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = [
        "| " + " | ".join([str(i)] + ["—" if pd.isna(v) else fmt.format(v) for v in row]) + " |"
        for i, row in zip(df.index, df.to_numpy(), strict=True)
    ]
    return "\n".join([head, sep, *body])


def render_pitch(case: Case, out_dir: str | Path, today: str | None = None) -> Path:
    """Write ``pitch.md`` and ``football_field.png`` into ``out_dir``; return the markdown path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_football_field(case, out / "football_field.png")
    c = case.company
    v = dcf_value(case)
    proj = project(case)
    g_imp = implied_growth(case)
    g_avg = sum(case.forecast.revenue_growth) / case.forecast.years
    y1 = year1_metrics(case)
    sens = sensitivity(case, (-0.01, 0.0, 0.01), (-0.005, 0.0, 0.005))
    sens.index = [f"WACC {r:.1%}" for r in sens.index]
    sens.index.name = ""
    sens.columns = [f"g {g:.1%}" for g in sens.columns]
    ff = football_field(case)
    ff.index.name = "method"

    title = f"# {c.name} ({c.ticker})" + (" — FICTIONAL EXAMPLE" if c.fictional else "")
    as_of = c.as_of or "n/a"
    lines = [
        title,
        "",
        f"**Price {c.price:,.2f} {c.currency}** ({as_of}) · base-case DCF "
        f"**{v.per_share:,.2f}** · **{valuation_label(v.upside)}**",
        "",
        f"_{DISCLAIMER}_",
        "",
        "## Thesis",
        _bullets(case.thesis, "No thesis written yet: add it to the case file."),
        "",
        "## Valuation",
        "![football field](football_field.png)",
        "",
        f"- **DCF (FCFF, {case.forecast.years}-year forecast):** EV {v.enterprise_value:,.0f}m, "
        f"equity {v.equity_value:,.0f}m, {v.per_share:,.2f} per share. "
        f"The terminal value is **{v.terminal_share:.0%}** of EV.",
        f"- **WACC {v.wacc:.1%}:** cost of equity {cost_of_equity(case):.1%} "
        f"(beta {case.discount.beta:g}), pre-tax cost of debt "
        f"{case.discount.pre_tax_cost_of_debt:.1%}, {case.discount.debt_weight:.0%} debt. "
        f"Terminal growth {v.terminal_growth:.1%}.",
        "- **Reverse DCF:** the price implies revenue growth of "
        + ("n/a" if pd.isna(g_imp) else f"**{g_imp:.1%} a year**")
        + f" for {case.forecast.years} years at these margins; the base case assumes "
        f"{g_avg:.1%} on average.",
        f"- **Year-1:** revenue {proj['revenue'].iloc[0]:,.0f}m, EBITDA {y1['ebitda']:,.0f}m, "
        f"EPS {y1['eps']:,.2f}.",
        "",
        _md_table(ff),
        "",
        "**Sensitivity (value per share):**",
        "",
        _md_table(sens),
        "",
        "## Catalysts",
        _bullets(case.catalysts, "None listed."),
        "",
        "## Risks",
        _bullets(case.risks, "None listed. Every case has risks; write them down."),
        "",
        "---",
        f"Generated {today or date.today().isoformat()} by quantskills.fundamental from the "
        f"case file. Sources: {c.source or 'not stated'}.",
    ]
    md = out / "pitch.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")  # LF on every OS
    return md
