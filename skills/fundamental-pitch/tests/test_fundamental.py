import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantskills.data.fundamentals import starter_case
from quantskills.fundamental import (
    Case,
    comps_range,
    comps_table,
    cost_of_equity,
    dcf_value,
    example_case,
    football_field,
    implied_growth,
    load_case,
    project,
    render_pitch,
    save_case,
    scenario_table,
    sensitivity,
    valuation_label,
    wacc,
)


def _flat_case(**over) -> Case:
    """One forecast year, no growth, no reinvestment: FCFF = NOPAT forever."""
    d = {
        "company": {"name": "Flat", "ticker": "FLAT", "price": 10.0, "shares_out": 10.0,
                    "net_debt": 0.0},
        "base": {"revenue": 100.0, "ebit_margin": 0.2, "tax_rate": 0.25, "da_pct_revenue": 0.0,
                 "capex_pct_revenue": 0.0, "nwc_pct_revenue": 0.0},
        "forecast": {"revenue_growth": [0.0], "ebit_margin": [0.2]},
        "discount": {"risk_free": 0.03, "equity_risk_premium": 0.05, "beta": 1.0,
                     "pre_tax_cost_of_debt": 0.05, "debt_weight": 0.0, "mid_year": False},
        "terminal": {"growth": 0.0},
    }  # fmt: skip
    for k, v in over.items():
        d[k] = {**d[k], **v}
    return Case.from_dict(d)


def test_wacc_is_capm_plus_after_tax_debt():
    c = _flat_case(discount={"debt_weight": 0.4})
    assert cost_of_equity(c) == pytest.approx(0.08)
    assert wacc(c) == pytest.approx(0.6 * 0.08 + 0.4 * 0.05 * (1 - 0.25))
    c2 = _flat_case(discount={"wacc_override": 0.1})
    assert wacc(c2) == 0.1


def test_zero_growth_dcf_is_a_perpetuity():
    c = _flat_case()
    nopat = 100 * 0.2 * 0.75  # 15
    v = dcf_value(c)
    assert v.enterprise_value == pytest.approx(nopat / 0.08)  # F / r
    assert v.per_share == pytest.approx(nopat / 0.08 / 10)
    assert v.upside == pytest.approx(v.per_share / 10 - 1)
    debt = dcf_value(_flat_case(company={"net_debt": 50.0}))
    assert debt.equity_value == pytest.approx(v.enterprise_value - 50)


def test_fcff_arithmetic():
    c = _flat_case(
        base={"da_pct_revenue": 0.05, "capex_pct_revenue": 0.08, "nwc_pct_revenue": 0.1},
        forecast={"revenue_growth": [0.1], "ebit_margin": [0.2]},
    )
    row = project(c).iloc[0]
    rev = 110.0
    expected = rev * 0.2 * 0.75 + rev * 0.05 - rev * 0.08 - (rev * 0.1 - 100 * 0.1)
    assert row["fcff"] == pytest.approx(expected)


def test_mid_year_raises_value():
    c = _flat_case(forecast={"revenue_growth": [0.05] * 3, "ebit_margin": [0.2] * 3})
    mid = replace(c, discount=replace(c.discount, mid_year=True))
    assert dcf_value(mid).per_share > dcf_value(c).per_share


def test_sensitivity_and_scenarios_move_the_right_way():
    c = example_case()
    s = sensitivity(c)
    assert (s.diff(axis=0).dropna() < 0).all().all()  # higher WACC, lower value
    assert (s.diff(axis=1).dropna(axis=1) > 0).all().all()  # higher growth, higher value
    st = scenario_table(c)
    assert st.loc["bear", "per_share"] < st.loc["base", "per_share"] < st.loc["bull", "per_share"]
    assert st.loc["base", "per_share"] == pytest.approx(dcf_value(c).per_share)


def test_reverse_dcf_reproduces_the_price():
    c = example_case()
    g = implied_growth(c)
    f = replace(c.forecast, revenue_growth=[g] * c.forecast.years)
    assert dcf_value(replace(c, forecast=f)).per_share == pytest.approx(c.company.price, rel=1e-4)
    assert np.isnan(implied_growth(c, price=1e6))  # no plausible growth gets there


def test_exit_multiple_and_errors():
    c = example_case()
    v = dcf_value(c, method="exit")
    last = project(c).iloc[-1]
    tv = 9.0 * (last["ebit"] + last["da"])
    assert v.pv_terminal == pytest.approx(tv / (1 + wacc(c)) ** 5)
    with pytest.raises(ValueError, match="exceed"):
        dcf_value(c, wacc_rate=0.02, terminal_growth=0.025)
    with pytest.raises(ValueError):
        dcf_value(c, method="magic")
    with pytest.raises(ValueError, match="exit"):
        dcf_value(_flat_case(), method="exit")


def test_comps_hand_check():
    c = example_case()
    t = comps_table(c)
    p1 = project(c).iloc[0]
    ebitda = p1["ebit"] + p1["da"]
    a = t.iloc[0]
    assert a["value_ev_ebitda"] == pytest.approx((8.5 * ebitda - 900) / 120)
    r = comps_range(c)
    assert (r["low"] <= r["mid"]).all() and (r["mid"] <= r["high"]).all()


@pytest.mark.parametrize(
    "section, field, value, msg",
    [
        ("forecast", "ebit_margin", [0.1], "per year"),
        ("company", "price", -1.0, "positive"),
        ("base", "tax_rate", 1.2, "tax_rate"),
        ("terminal", "growth", 0.08, "GDP"),
    ],
)
def test_validation(section, field, value, msg):
    d = example_case().to_dict()
    d[section][field] = value
    with pytest.raises(ValueError, match=msg):
        Case.from_dict(d)


def test_missing_sections_and_round_trip(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        Case.from_dict({"company": {}})
    c = example_case()
    save_case(c, tmp_path / "c.json")
    assert load_case(tmp_path / "c.json") == c


def test_pitch_lays_out_but_never_writes_the_thesis(tmp_path):
    c = example_case()
    md = render_pitch(c, tmp_path, today="2026-01-01").read_text(encoding="utf-8")
    assert "FICTIONAL" in md and "Not investment advice" in md
    assert c.thesis[0] in md
    assert (tmp_path / "football_field.png").stat().st_size > 1000
    blank = replace(c, thesis=[], risks=[])
    md2 = render_pitch(blank, tmp_path / "b").read_text(encoding="utf-8")
    assert "No thesis written yet" in md2 and "write them down" in md2
    ff = football_field(c)
    assert {"low", "mid", "high"} <= set(ff.columns) and (ff["low"] <= ff["high"]).all()
    assert valuation_label(0.3).startswith("Base-case value is 30% above")
    assert "within" in valuation_label(0.05)


def _fake_yahoo(_ticker: str) -> dict:
    cols = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])
    inc = pd.DataFrame(
        {
            c: v
            for c, v in zip(
                cols,
                [
                    [1000e6, 150e6, 140e6, 30e6],
                    [1100e6, 170e6, 160e6, 35e6],
                    [1200e6, 190e6, 180e6, 40e6],
                ],  # fmt: skip
                strict=True,
            )
        },
        index=["Total Revenue", "Operating Income", "Pretax Income", "Tax Provision"],
    )
    cf = pd.DataFrame(
        {c: [-60e6, 40e6] for c in cols}, index=["Capital Expenditure", "Depreciation"]
    )
    info = {"longName": "Fake Co", "currentPrice": 20.0, "sharesOutstanding": 50e6,
            "totalDebt": 200e6, "totalCash": 50e6, "beta": 1.2, "currency": "USD"}  # fmt: skip
    return {"info": info, "income": inc, "balance": pd.DataFrame(), "cashflow": cf}


def test_starter_case_from_reported_data(tmp_path):
    d = starter_case("fake", downloader=_fake_yahoo, cache_dir=tmp_path)
    c = Case.from_dict(d)
    assert c.base.revenue == pytest.approx(1200.0)
    assert c.base.ebit_margin == pytest.approx(190 / 1200)
    assert c.company.net_debt == pytest.approx(150.0)
    assert c.thesis == [] and not c.company.fictional
    assert json.loads((tmp_path / "FAKE.json").read_text())["company"]["ticker"] == "FAKE"
    bad = _fake_yahoo("x")
    bad["income"] = bad["income"].drop(index="Total Revenue")
    with pytest.raises(KeyError, match="revenue"):
        starter_case("x", downloader=lambda _t: bad, cache_dir=tmp_path)
