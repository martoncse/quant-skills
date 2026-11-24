---
name: portfolio-risk
description: Build portfolios (min-variance, mean-variance, HRP, PCA eigenportfolios) and measure their risk (VaR, CVaR, risk contributions).
---

## When to use
To turn a universe of return series into weights without (or with) return
forecasts, and to report how much a portfolio could lose. Strategy #7 and the
terminal's RISK screen are built on this skill.

## Inputs / outputs
All inputs are wide `date x ticker` **simple daily returns**; weights are `pd.Series` by ticker.
- Covariance: `sample_cov(r)`, `ledoit_wolf_cov(r)` (annualised), `cov_to_corr(cov)`.
- Allocators (weights sum to 1):
  - `equal_weight(tickers)`, `inverse_volatility(cov)`
  - `min_variance(cov, long_only, max_weight)`
  - `mean_variance(mu, cov, risk_aversion)`, `max_sharpe(mu, cov)`
  - `hrp_weights(cov)`, plus `cluster_order(corr)` to see the clusters
- PCA: `pca_factors(r) -> PCAResult(eigenvalues, explained, loadings, vol)` and
  `eigenportfolio(r, k)` (unit gross).
- Dispatcher: `allocate(r, method)` with `method` in `ALLOCATORS`.
- Risk (1-day, **positive = loss**):
  - `historical_var`, `historical_cvar`
  - `parametric_var` (Gaussian), `cornish_fisher_var` (adjusted for skew and kurtosis)
  - `risk_contributions(w, cov)` (sums to 1)
  - `risk_report(r, w) -> dict`
- Snapshot: `risk_view(prices, method, lookback) -> RiskView` (weights table, metrics,
  clustered correlation). Used by the terminal.

## Steps
1. `r = simple_returns(to_wide(prices)).iloc[-252:]`
2. `w = allocate(r, "hrp")`. Compare it with `"min_var"` and `"equal"`.
3. `risk_report(r, w)` gives VaR/CVaR at 95% and 99%; `risk_contributions(w, ledoit_wolf_cov(r))`.
4. CLI: `uv run python skills/portfolio-risk/scripts/risk_report.py --method hrp`
5. Terminal: `RISK hrp 252` (or F4).

## Pitfalls
- **Estimation error dominates.** Mean-variance treats noisy `mu` as truth and piles
  into a few names. Prefer min-variance, HRP or strong shrinkage unless you trust `mu`.
- **Shrink the covariance** when there are few days per asset (Ledoit-Wolf by default).
- **VaR is a quantile, not a worst case.** CVaR says how bad the tail is. Historical
  VaR only knows the window it saw. Gaussian VaR understates fat tails; Cornish-Fisher
  helps a little.
- In-window risk of today's weights is **in-sample**: the weights were chosen on the same data.
- Correlations rise in crises. Diversification measured in calm periods overstates protection.
- Survivorship: a universe of today's survivors understates historical risk.

## References
- ML4T 3rd ed., ch. 14 (PCA / latent factors), ch. 17 (portfolio construction, HRP), ch. 19 (risk)
- Markowitz (1952); Ledoit & Wolf (2004); López de Prado (2016), "Building Diversified
  Portfolios that Outperform Out of Sample"; Avellaneda & Lee (2010) on eigenportfolios
- PyPortfolioOpt and Riskfolio-Lib: design ideas only. Everything is written with scipy/numpy.
