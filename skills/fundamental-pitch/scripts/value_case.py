r"""Value a case file with a DCF and comps, and optionally write the one-page pitch.

uv run python skills/fundamental-pitch/scripts/value_case.py                 # bundled example
uv run python skills/fundamental-pitch/scripts/value_case.py my_case.json --pitch out/
uv run python skills/fundamental-pitch/scripts/value_case.py --fetch MSFT    # starter case

--fetch downloads the latest annual statements from Yahoo Finance into
data/fundamentals/<TICKER>.json: a *draft* with the base year filled in. Edit the
forecast, discount rate, peers, scenarios and thesis before trusting any value.
"""

import argparse

import pandas as pd

from quantskills.data.fundamentals import starter_case
from quantskills.fundamental import (
    Case,
    comps_table,
    dcf_value,
    example_case,
    implied_growth,
    load_case,
    project,
    render_pitch,
    scenario_table,
    sensitivity,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("case", nargs="?", help="case JSON (default: the fictional example)")
    p.add_argument("--pitch", metavar="DIR", help="write pitch.md + football_field.png here")
    p.add_argument("--fetch", metavar="TICKER", help="build a starter case from Yahoo Finance")
    a = p.parse_args()

    if a.fetch:
        d = starter_case(a.fetch)
        print(f"wrote data/fundamentals/{a.fetch.upper()}.json: edit its assumptions, then run")
        print(f"  uv run python {__file__} data/fundamentals/{a.fetch.upper()}.json")
        case = Case.from_dict(d)
    else:
        case = load_case(a.case) if a.case else example_case()

    c = case.company
    pd.set_option("display.width", 120)
    print(f"{c.name} ({c.ticker}){'  [FICTIONAL]' if c.fictional else ''}  price {c.price:,.2f}\n")
    print(project(case).round(1).to_string(), "\n")
    v = dcf_value(case)
    print(
        f"DCF: EV {v.enterprise_value:,.0f}m, equity {v.equity_value:,.0f}m, "
        f"{v.per_share:,.2f}/share ({v.upside:+.0%} vs price); WACC {v.wacc:.2%}, "
        f"g {v.terminal_growth:.1%}, terminal value {v.terminal_share:.0%} of EV"
    )
    g = implied_growth(case)
    print(f"Reverse DCF: price implies {g:.1%} revenue growth a year\n" if g == g else "")
    print("Sensitivity (per share), WACC x terminal growth:")
    print(sensitivity(case).round(2).to_string(), "\n")
    if case.scenarios:
        print(scenario_table(case).round(3).to_string(), "\n")
    if case.peers:
        print(comps_table(case).round(2).to_string(index=False), "\n")
    if a.pitch:
        print(f"pitch: {render_pitch(case, a.pitch)}")


if __name__ == "__main__":
    main()
