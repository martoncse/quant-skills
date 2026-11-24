---
name: models
description: Fit and cross-validate ML return models on (date, ticker) panels with purged, embargoed CV; LightGBM with SHAP.
---

## When to use
When a strategy learns a mapping from features to future returns (strategies #2
and #6), and you need to choose hyperparameters without leaking the future.

## Inputs / outputs
- `feature_panel(close, names) -> DataFrame`: long `(date, ticker)`, one column per
  feature, z-scored across tickers each day (`features/panel.py`, `ML_FEATURES`).
- `label_panel(close, horizon, kind) -> Series`: forward return as `"raw"`, `"rank"`
  (percentile - 0.5) or `"up"` (above the median). The last `horizon` dates are NaN.
- `dropna_xy(X, y)`: aligned, complete rows.
- `purged_kfold(dates, n_splits, horizon, embargo)` and
  `walk_forward_splits(dates, n_splits, horizon)` (`utils/splits.py`) yield
  `(train_dates, test_dates)`.
- `cv_score(make_model, X, y, ...) -> list[float]`: per-fold mean daily Spearman IC.
- `fit_lgbm_cv(X, y, params, max_rounds, ...) -> (booster, report)`: rounds chosen
  by early stopping per purged fold (median), then refit on all rows.
- `shap_importance(booster, X) -> Series`: mean |SHAP| via LightGBM's TreeSHAP.

## Steps
1. Build `X` and `y` from the **training window only**: `dropna_xy(feature_panel(c), label_panel(c, 21))`.
2. Score candidates with `cv_score` (or `fit_lgbm_cv`) using `horizon` equal to
   the label horizon. Pick the best by mean IC.
3. Refit on the whole training window, predict every date, and rank into a book
   (`strategies/_panel_ml.py` does all of this).
4. CLI: `uv run python skills/models/scripts/cv_report.py --model ridge`

## Pitfalls
- **Never shuffle time series.** Neighbouring days share 20 of 21 days of a 21-day
  label. With shuffled K-fold, the model sees its test labels in training. The demo
  shows shuffled CV reporting IC ≈ 0.25 on **pure noise**; purged CV reports ≈ 0.
- **Purge = label horizon.** A too-small `horizon` in the splitter leaks again.
- **Embargo** covers serial correlation in features after the test block.
- Early stopping on the CV test fold makes the CV IC slightly optimistic. Use it
  to *choose rounds*, and judge the model on the untouched hold-out.
- Every model, grid point and feature set you try is a trial. Pass the count to the DSR.
- In-sample predictions of a fitted model are not a backtest. Only the hold-out is.

## References
- ML4T 3rd ed., ch. 6–7 (purged CV, labels), ch. 11 (linear models), ch. 12 (gradient boosting, SHAP)
- López de Prado (2018), *Advances in Financial ML*, ch. 7 (purged k-fold, embargo)
- Lundberg & Lee (2017), SHAP; Ke et al. (2017), LightGBM
