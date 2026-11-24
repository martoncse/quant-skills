"""Relative valuation from peer multiples (EV/EBITDA and P/E) on forecast year 1.

    implied EV      = peer EV/EBITDA x EBITDA_1          -> per share = (EV - net debt) / shares
    implied price   = peer P/E x EPS_1
    EPS_1           = (EBIT_1 - interest) (1 - tax) / shares,  interest = r_d x max(net debt, 0)

Peers should be genuinely comparable (business, growth, margins, leverage). The range,
not the median, is the honest summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantskills.fundamental.case import Case
from quantskills.fundamental.dcf import project


def year1_metrics(case: Case) -> dict[str, float]:
    p = project(case).iloc[0]
    interest = case.discount.pre_tax_cost_of_debt * max(case.company.net_debt, 0.0)
    net_income = (p["ebit"] - interest) * (1.0 - case.base.tax_rate)
    return {
        "ebitda": float(p["ebit"] + p["da"]),
        "net_income": float(net_income),
        "eps": float(net_income / case.company.shares_out),
    }


def comps_table(case: Case) -> pd.DataFrame:
    """One row per peer: its multiples and the per-share value each implies for the company."""
    m = year1_metrics(case)
    c = case.company
    rows = []
    for peer in case.peers:
        row: dict[str, object] = {"peer": peer.name, "ev_ebitda": peer.ev_ebitda, "pe": peer.pe}
        row["value_ev_ebitda"] = (
            (peer.ev_ebitda * m["ebitda"] - c.net_debt) / c.shares_out
            if peer.ev_ebitda is not None
            else np.nan
        )
        row["value_pe"] = peer.pe * m["eps"] if peer.pe is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def comps_range(case: Case) -> pd.DataFrame:
    """Low / median / high value per share for each multiple (25th, 50th, 75th percentile)."""
    t = comps_table(case)
    out = {}
    for col, label in (("value_ev_ebitda", "EV/EBITDA comps"), ("value_pe", "P/E comps")):
        v = t[col].dropna().to_numpy(dtype=float) if col in t else np.array([])
        if len(v):
            out[label] = {
                "low": float(np.quantile(v, 0.25)),
                "mid": float(np.median(v)),
                "high": float(np.quantile(v, 0.75)),
            }
    return pd.DataFrame(out).T
