---
name: features-alpha
description: Build look-ahead-free cross-sectional alpha factors (momentum, reversal, low-vol) and forward-return labels, then check them with IC.
---

## When to use
When creating or testing a predictive signal for a cross-section of assets,
before it goes into a model or strategy.

## Inputs / outputs
- Input: wide `close` frame (`date x ticker`), from `to_wide(prices)`.
- Factors (row *t* uses rows <= *t* only):
  - `momentum(close, lookback=252, skip=21)`: P(t-skip) / P(t-lookback) - 1
  - `reversal(close, window=21)`: -(P(t) / P(t-window) - 1)
  - `low_volatility(close, window=63)`: minus trailing daily-return std
- `compute_factors(close, names)` returns `{name: DataFrame}`.
- `zscore_cross_section`, `rank_cross_section`: per-date normalisation.
- Label: `forward_returns(close, horizon)`, the return from *t* to *t+h*, stored at *t*.

## Steps
1. `factors = compute_factors(close)`
2. `fwd = forward_returns(close, 21)`
3. `ic = information_coefficient(factors["momentum"], fwd)` then `ic_summary(ic, horizon=21)`
4. `quantile_returns(factor, fwd)` should rise across quintiles if the factor works.
5. Every new factor gets a test: `assert_no_lookahead(my_factor, close)`.
   It scrambles all rows after *t* and checks the values at <= *t* don't change.
6. Script: `uv run python skills/features-alpha/scripts/factor_report.py`

## Pitfalls
- **Shift labels, not features.** `forward_returns` is the only forward-looking
  function, and it must never be a model input.
- Overlapping horizons inflate t-stats. `ic_summary(horizon=h)` samples
  every *h*-th day for the t-stat.
- Cross-sectional z-scores need enough names per date. Below about 10, ranks are noisy.
- Value and quality are deferred: they need point-in-time fundamentals. Using
  today's fundamentals for past dates is look-ahead.

## References
- ML4T 3rd ed., ch. 7 (labels, IC, quantile analysis), ch. 8 (price features)
- Jegadeesh & Titman (1993) momentum; Ang et al. (2006) low volatility
- alphalens-reloaded (Apache-2.0) for the IC/quantile tear-sheet idea
