---
name: robustness
description: Test whether a backtest's edge is real: deflated Sharpe, PBO (CSCV), noise/delay/parameter perturbations, regime splits and forecast calibration, graded red/amber/green.
---

## When to use
After a strategy backtests well, and **before** you believe it. Use it to ask:
was the winning configuration picked from noise (DSR, PBO)? Does the edge survive
small changes it should not care about (price noise, a day's delay, nearby params)?
Does it only work in one kind of market (regimes)? Are its probabilities honest
(calibration)? The terminal's `ROBUST` screen (F5) runs the same report.

## Inputs / outputs
- Prices: long `MultiIndex(date, ticker)` OHLCV; a `train_end` date; a strategy name or instance.
- `robustness_report(strategy, prices, train_end, mode="walk-forward") -> RobustnessReport`:
  - `.checks` (list of `Check(name, value, display, rule, status)`), `.verdict`
    (`green` / `amber` / `red`), `.checks_table()`, `.to_markdown()`
  - `.dsr` (`DeflatedSharpe`), `.pbo` (`PBOResult` or None), `.surface` (2-D Sharpe grid)
  - `.noise` / `.noise_summary()`, `.delay`, `.regimes`
  - `.is_sharpe`, `.wf_sharpe` (walk-forward inside the training window), `.oos_sharpe`
- Building blocks:
  - `pbo_cscv(trial_returns T x N, n_blocks=16) -> PBOResult(pbo, logits, prob_oos_loss, ...)`
  - `add_price_noise(prices, bps, seed)`, `delay_weights(w, days)`, `noise_test`, `delay_test`
  - `sensitivity_surface(cls, params, {p1: [...], p2: [...]}, train)` -> `(surface, trial_returns)`
  - `walk_forward_backtest(cls, params, train, n_folds=4)` (in `robustness.perturb`)
  - `regime_labels(close, "vol" | "trend")`, `regime_table(returns, labels)`, `regime_report`
  - `brier_score`, `log_loss`, `reliability_table`, `expected_calibration_error`,
    `brier_decomposition`, `calibration_summary`; out-of-fold probabilities from
    `quantskills.models.oof_predictions`
  - Figures: `plot_pbo`, `plot_surface`, `plot_decay`, `plot_reliability`
- Thresholds: `THRESHOLDS` in `robustness/verdict.py` (one place, easy to audit).

## Steps
1. `train_end = holdout_date(dates, 0.3)`; decide the configuration *before* looking at the hold-out.
2. `r = robustness_report("pairs-coint", prices, train_end)`; `print(r.to_markdown())`.
3. Read the walk-forward line first: in-sample Sharpe → walk-forward Sharpe. A big
   drop means the model memorised its training data.
4. DSR and PBO say whether the choice among configs is luck; the noise, delay and
   surface checks say whether the edge is fragile.
5. Read the hold-out row **last, once**. Don't re-tune after reading it.
6. CLI: `uv run python skills/robustness/scripts/robustness_report.py --all --figures`;
   calibration: `.../calibration_report.py`. Terminal: `ROBUST pairs-coint` or F5.

## Pitfalls
- **Scoring on the fitted days hides overfitting.** That's why checks default to
  walk-forward. With `mode="in-sample"`, LightGBM passes everything on synthetic
  data and then fails the hold-out.
- **PBO measures the selection, not profitability.** On a plateau where every config
  works, the IS winner's OOS rank is a coin flip, so PBO ≈ 0.5 while P(OOS loss) ≈ 0.
  The check grades on the better of the two.
- **Retention ratios need a real base.** Below a base Sharpe of 0.5 the
  noise/delay/surface checks show `n/a`. A weak strategy is already flagged by DSR.
- **The grid is small (3 x 3).** PBO with 9 configs is coarse; DSR counts those 9
  as trials. Report every config you tried before, via `n_trials`.
- **Regimes on short samples are noisy.** Regimes with fewer than 63 days are not scored.
- **Price noise is traded at clean prices.** On noisy prices a reversal rule would
  "profit" from the noise itself.
- Calibration matters only when probabilities are used as probabilities (sizing,
  confidence). A ranking strategy can be useful and badly calibrated.
- **Not yet covered:** adversarial stress on text/sentiment signals arrives with
  strategy #8 (Phase 5).
- Green is a triage flag, not proof. Synthetic results illustrate the method only.

## References
- Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of Backtest Overfitting" (CSCV/PBO)
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7, 11-14
- Brier (1950); Murphy (1973); Guo et al. (2017) on calibration
- ML4T 3rd ed.: backtesting pitfalls and strategy evaluation chapters (see docs/references.md)
- Write-up with figures: [docs/robustness.md](../../docs/robustness.md)
