---
name: fundamental-pitch
description: Value a company with a DCF (FCFF, WACC via CAPM), peer multiples and scenarios, then lay out a one-page pitch from a case file.
---

## When to use
To value a single company from its fundamentals, stress the valuation (sensitivity,
scenarios, reverse DCF), and present it on one page. Complements the quant strategies:
this skill is about one stock's value, not a signal across many.

## Inputs / outputs
- **Case file** (JSON, see `src/quantskills/fundamental/examples/example_industrial.json`):
  - `company`: price, diluted shares (m), net debt (m), currency, `fictional`, `source`
  - `base`: last-year revenue (m), EBIT margin, tax rate, D&A / capex / NWC as % of revenue
  - `forecast`: one revenue growth and one EBIT margin per year
  - `discount`: risk-free, ERP, beta, pre-tax cost of debt, debt weight (or `wacc_override`)
  - `terminal`: perpetual growth, optional exit EV/EBITDA
  - `peers` (EV/EBITDA, P/E), `scenarios` (shifts to growth, margin, WACC, terminal g)
  - `thesis`, `risks`, `catalysts`: **your own words**; the pitch never writes them
- `load_case(path)`, `example_case()` -> `Case` (validated on load).
- `dcf_value(case, wacc_rate, terminal_growth, method="gordon"|"exit") -> Valuation`
  (EV, equity, per share, upside, terminal share of EV).
- `project(case)` (yearly FCFF build), `wacc`, `cost_of_equity`, `sensitivity` (WACC x g),
  `scenario_table`, `implied_growth` (reverse DCF), `comps_table`, `comps_range`.
- `football_field(case)`, `plot_football_field(case, path)`, `render_pitch(case, dir)` ->
  `pitch.md` + `football_field.png`.
- `quantskills.data.fundamentals.starter_case(ticker)`: a draft case from Yahoo Finance
  (network, cached in `data/fundamentals/`).

## Steps
1. Start from the example or `value_case.py --fetch TICKER` (a draft: base year only).
2. Write the forecast: growth and margins year by year, each with a reason.
3. Set the discount rate (CAPM inputs, debt weight) and terminal growth (≤ long-run GDP).
4. Add 4–8 genuinely comparable peers and bear / bull scenarios.
5. `uv run python skills/fundamental-pitch/scripts/value_case.py case.json --pitch out/`
6. Read the reverse DCF first: what growth does the price already assume?
7. Terminal: `DCF` (the example) or `DCF path/to/case.json`.

## Pitfalls
- **The terminal value dominates.** It is often 60–80% of EV, so WACC and *g* drive the
  answer. Always show the sensitivity grid; never one number.
- **g must stay below WACC and below long-run nominal GDP** (the case rejects g > 5%).
- **Consistency:** growth needs reinvestment. High growth with falling capex is a red flag.
- **Net debt and shares:** use diluted shares; include leases and pensions in debt if material.
- **Comps:** a cheap peer may be cheap for a reason. The range matters more than the median.
- **Point-in-time:** use statements that were public at the valuation date. Yahoo data is
  an unaudited aggregation; check the filings.
- **Narrative honesty:** the bundled example is fictional and labelled so. Never present
  invented numbers as a real company's.

## References
- Damodaran, *Investment Valuation* (3rd ed.); Koller, Goedhart & Wessels, *Valuation* (7th ed.)
- Rappaport & Mauboussin, *Expectations Investing* (reverse DCF)
- Demo: `demo.ipynb`; example output: `examples/example_industrial/pitch.md`
