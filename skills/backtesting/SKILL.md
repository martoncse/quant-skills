---
name: backtesting
description: Backtest target-weight strategies with costs on, drift between rebalances, and a fit-on-train / test-on-holdout runner.
---

## When to use
To turn a `Strategy`'s target weights into a net-of-cost return series, and to
run the standard fit-on-train, evaluate-on-holdout experiment.

## Inputs / outputs
- `run_backtest(target_weights, close, costs=CostModel()) -> BacktestResult`
  - `target_weights`: `date x ticker`, fractions of NAV (negative = short).
    An all-NaN row means "don't rebalance; let positions drift".
  - `CostModel(commission_bps=5, slippage_bps=5)` is non-zero by default.
  - `BacktestResult`: `returns` (net), `gross_returns`, `costs`, `turnover`,
    `weights` (post-trade), `equity`, `.split(date)`.
- `Strategy` interface (`src/quantskills/strategies/base.py`):
  `fit(data) -> self`, `signals(data) -> weights`, `params: dict`.
- `run_strategy(strategy, prices, train_end, costs) -> StrategyRun` with
  `.full`, `.in_sample`, `.out_of_sample`.
- `sweep_params(cls, grid, prices, train_end)`: in-sample-only parameter sweep.

## Steps
1. `train_end = holdout_date(dates, test_fraction=0.3)`
2. `run = run_strategy(FactorLongShort(), prices, train_end)`
3. `tear_sheet(run.in_sample)`. Iterate here only.
4. When you're done iterating, look at `tear_sheet(run.out_of_sample, n_trials=N)` once.
5. CLI: `uv run python skills/backtesting/scripts/run_backtest.py --plot out.png`

## Timing convention
Weights in row *t* are decided at the close of *t*, traded at that close, and
earn returns from *t* to *t+1*. Turnover is measured against the *drifted*
book: `cost_t = sum|w_target - w_drifted| * (commission + slippage) / 1e4`.

## Pitfalls
- Same-bar execution: if a signal uses the close of *t*, trading at that close
  is optimistic. Add slippage, or shift the signal one day for open-to-open realism.
- Costs scale with turnover. Check `gross_sharpe` against `sharpe` and `ann_cost_drag`.
- Linear costs ignore market impact, borrow fees for shorts and capacity.
- `fit` must only see training data. The runner enforces this, so don't bypass it.
- Running many configs and reporting the best is selection bias. Count the trials (see `evaluation`).

## References
- ML4T 3rd ed., ch. 6 (research framework), ch. 16 (strategy simulation),
  ch. 18 (transaction costs)
- vectorbt and backtrader (design ideas only; no code used)
