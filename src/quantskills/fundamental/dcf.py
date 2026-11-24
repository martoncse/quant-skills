"""Discounted cash flow valuation (free cash flow to the firm).

    FCFF_t = EBIT_t (1 - tax) + D&A_t - capex_t - change in NWC_t
    EV     = sum_t FCFF_t / (1 + WACC)^(t - 0.5)  +  TV / (1 + WACC)^N      (mid-year)
    TV     = FCFF_{N+1} / (WACC - g)                                        (Gordon growth)
    equity = EV - net debt;   value per share = equity / diluted shares

    WACC = E/V * (r_f + beta * ERP) + D/V * r_d * (1 - tax)                 (CAPM for equity)

References: Damodaran, *Investment Valuation* (3rd ed.); Koller, Goedhart & Wessels,
*Valuation* (McKinsey, 7th ed.).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from quantskills.fundamental.case import Case, Forecast


def cost_of_equity(case: Case) -> float:
    d = case.discount
    return d.risk_free + d.beta * d.equity_risk_premium


def wacc(case: Case) -> float:
    d = case.discount
    if d.wacc_override is not None:
        return d.wacc_override
    kd_after_tax = d.pre_tax_cost_of_debt * (1.0 - case.base.tax_rate)
    return (1.0 - d.debt_weight) * cost_of_equity(case) + d.debt_weight * kd_after_tax


def project(case: Case) -> pd.DataFrame:
    """Year-by-year forecast: revenue, EBIT, NOPAT, D&A, capex, change in NWC, FCFF."""
    b, f = case.base, case.forecast
    da = f.da_pct_revenue if f.da_pct_revenue is not None else b.da_pct_revenue
    capex = f.capex_pct_revenue if f.capex_pct_revenue is not None else b.capex_pct_revenue
    nwc = f.nwc_pct_revenue if f.nwc_pct_revenue is not None else b.nwc_pct_revenue
    rows = []
    rev_prev, nwc_prev = b.revenue, b.revenue * b.nwc_pct_revenue
    for t, (g, m) in enumerate(zip(f.revenue_growth, f.ebit_margin, strict=True), start=1):
        rev = rev_prev * (1.0 + g)
        ebit = rev * m
        nopat = ebit * (1.0 - b.tax_rate)
        nwc_t = rev * nwc
        fcff = nopat + rev * da - rev * capex - (nwc_t - nwc_prev)
        rows.append(
            {
                "year": t,
                "revenue": rev,
                "growth": g,
                "ebit": ebit,
                "ebit_margin": m,
                "nopat": nopat,
                "da": rev * da,
                "capex": rev * capex,
                "delta_nwc": nwc_t - nwc_prev,
                "fcff": fcff,
            }
        )
        rev_prev, nwc_prev = rev, nwc_t
    return pd.DataFrame(rows).set_index("year")


@dataclass(frozen=True)
class Valuation:
    enterprise_value: float
    equity_value: float
    per_share: float
    upside: float  # per_share / price - 1
    pv_fcff: float
    pv_terminal: float
    terminal_share: float  # share of EV from the terminal value
    wacc: float
    terminal_growth: float
    method: str


def dcf_value(
    case: Case,
    wacc_rate: float | None = None,
    terminal_growth: float | None = None,
    method: str = "gordon",
) -> Valuation:
    """Value per share. ``method``: ``"gordon"`` (perpetual growth) or ``"exit"`` (EV/EBITDA)."""
    r = wacc(case) if wacc_rate is None else wacc_rate
    g = case.terminal.growth if terminal_growth is None else terminal_growth
    proj = project(case)
    n = len(proj)
    t = proj.index.to_numpy(dtype=float)
    timing = t - 0.5 if case.discount.mid_year else t
    pv_fcff = float((proj["fcff"] / (1.0 + r) ** timing).sum())
    last = proj.iloc[-1]
    if method == "gordon":
        if r <= g:
            raise ValueError(f"WACC ({r:.2%}) must exceed terminal growth ({g:.2%})")
        # Year N+1 cash flow at steady state: reinvestment that supports growth g.
        fcff_next = last["fcff"] * (1.0 + g)
        tv = fcff_next / (r - g)
    elif method == "exit":
        multiple = case.terminal.exit_ev_ebitda
        if multiple is None:
            raise ValueError("exit method needs terminal.exit_ev_ebitda")
        tv = multiple * (last["ebit"] + last["da"])
    else:
        raise ValueError("method must be 'gordon' or 'exit'")
    tv_timing = (n - 0.5) if case.discount.mid_year and method == "gordon" else n
    pv_tv = float(tv / (1.0 + r) ** tv_timing)
    ev = pv_fcff + pv_tv
    equity = ev - case.company.net_debt
    per_share = equity / case.company.shares_out
    return Valuation(
        enterprise_value=ev,
        equity_value=equity,
        per_share=per_share,
        upside=per_share / case.company.price - 1.0,
        pv_fcff=pv_fcff,
        pv_terminal=pv_tv,
        terminal_share=pv_tv / ev if ev else float("nan"),
        wacc=r,
        terminal_growth=g,
        method=method,
    )


def sensitivity(
    case: Case,
    wacc_deltas: Sequence[float] = (-0.01, -0.005, 0.0, 0.005, 0.01),
    growth_deltas: Sequence[float] = (-0.01, -0.005, 0.0, 0.005, 0.01),
) -> pd.DataFrame:
    """Value per share over WACC (rows) x terminal growth (columns), around the base case."""
    r0, g0 = wacc(case), case.terminal.growth
    rows = [r0 + d for d in wacc_deltas]
    cols = [g0 + d for d in growth_deltas]
    out = pd.DataFrame(np.nan, index=pd.Index(rows, name="wacc"), columns=cols)
    out.columns.name = "terminal_growth"
    for r in rows:
        for g in cols:
            if r > g:
                out.loc[r, g] = dcf_value(case, r, g).per_share
    return out


def apply_scenario(case: Case, name: str) -> Case:
    s = case.scenarios[name]
    f = case.forecast
    forecast = Forecast(
        revenue_growth=[g + s.growth_shift for g in f.revenue_growth],
        ebit_margin=[m + s.margin_shift for m in f.ebit_margin],
        capex_pct_revenue=f.capex_pct_revenue,
        da_pct_revenue=f.da_pct_revenue,
        nwc_pct_revenue=f.nwc_pct_revenue,
    )
    discount = replace(case.discount, wacc_override=wacc(case) + s.wacc_shift)
    terminal = replace(
        case.terminal,
        growth=case.terminal.growth if s.terminal_growth is None else s.terminal_growth,
    )
    return replace(case, forecast=forecast, discount=discount, terminal=terminal)


def scenario_table(case: Case) -> pd.DataFrame:
    """Value per share and upside for ``base`` and each named scenario."""
    rows = {"base": dcf_value(case)}
    for name in case.scenarios:
        rows[name] = dcf_value(apply_scenario(case, name))
    return pd.DataFrame(
        {
            k: {
                "per_share": v.per_share,
                "upside": v.upside,
                "wacc": v.wacc,
                "terminal_growth": v.terminal_growth,
                "terminal_share": v.terminal_share,
            }
            for k, v in rows.items()
        }
    ).T


def implied_growth(case: Case, price: float | None = None) -> float:
    """Reverse DCF: the constant revenue growth (all forecast years) that justifies ``price``.

    Margins, WACC and terminal growth stay as in the case. NaN when no growth in
    [-50%, +100%] a year gets there.
    """
    target = case.company.price if price is None else price
    n = case.forecast.years

    def gap(g: float) -> float:
        f = replace(case.forecast, revenue_growth=[g] * n)
        return dcf_value(replace(case, forecast=f)).per_share - target

    lo, hi = -0.5, 1.0
    if np.sign(gap(lo)) == np.sign(gap(hi)):
        return float("nan")
    return float(brentq(gap, lo, hi, xtol=1e-6))
