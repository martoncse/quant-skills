---
name: evaluation
description: Evaluate backtests honestly with Sharpe/drawdown metrics, cost drag, factor IC, and the deflated Sharpe ratio over all trials.
---

## When to use
After every backtest or parameter sweep, and before quoting any number in a
README, PR or write-up.

## Inputs / outputs
- `summary(returns) -> dict`: ann. return/vol, Sharpe, Sortino, max DD, Calmar, hit rate
- `tear_sheet(result, n_trials=1, trial_sharpes=None) -> dict`: summary plus
  gross Sharpe, turnover, cost drag, PSR and DSR. Leading flat warm-up days are dropped.
- `format_tear_sheet(sheet)` returns text; `plot_tear_sheet(result, path, split_date=...)` saves a PNG.
- `deflated_sharpe_ratio(returns, n_trials, trial_sharpes) -> DeflatedSharpe`
  with `sharpe`, `sharpe_threshold`, `psr`, `dsr`.
- `information_coefficient`, `ic_summary`, `quantile_returns` for factors.
- `bayesian_sharpe(returns, n_draws, credible=0.9, seed) -> BayesianSharpe`: exact
  posterior of the annualised Sharpe (normal returns, flat prior) with `mean`,
  `ci_low`, `ci_high`, `prob_positive` and `draws`.

## Steps
1. Sweep configurations on the training window: `sweep_params(...)`.
2. Pick the best and compute `deflated_sharpe_ratio(is_returns, trial_sharpes=table["is_sharpe"])`.
3. Read the hold-out once and report it with the same `trial_sharpes`.
4. CLI: `uv run python skills/evaluation/scripts/select_and_deflate.py`
5. All strategies, one table: `uv run python skills/evaluation/scripts/strategy_gallery.py`

## Reading the deflated Sharpe
- PSR is the probability the true Sharpe is above 0, given sample length, skew and fat tails.
- DSR is the probability the true Sharpe beats the Sharpe you'd expect from the
  best of N *useless* strategies. The threshold grows with N and with how much
  the trial Sharpes vary.
- Rule of thumb: DSR < 0.95 means the result is consistent with luck from the search.

## Pitfalls
- Report **every** trial, including abandoned ideas. N is usually larger than you think.
- Without `trial_sharpes`, the DSR falls back to the Sharpe estimator's own
  variance. That is a lower bound on the penalty.
- The IC t-stat with overlapping horizons must pass `horizon=` to `ic_summary`.
- Never present backtests as expected returns; `format_tear_sheet` adds that disclaimer.
- The Bayesian Sharpe interval assumes i.i.d. normal returns. With fat tails and
  autocorrelation the real uncertainty is wider, so treat it as a lower bound.

## References
- ML4T 3rd ed., ch. 7 (IC, DSR), ch. 16 (overfitting tests)
- Bailey & López de Prado (2012, 2014): PSR, DSR
- López de Prado, *Advances in Financial ML*, ch. 11–14
- quantstats (Apache-2.0) for the tear-sheet layout idea
