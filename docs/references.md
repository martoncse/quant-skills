# References

## Machine Learning for Trading, 3rd edition (Stefan Jansen)
Chapter list checked against github.com/stefan-jansen/machine-learning-for-trading
on 2026-11-23. Ideas only. No code or prose is copied.

| # | Strategy (CLAUDE.md) | Main chapter(s) | Supporting |
|---|---|---|---|
| 1 | Factor long-short | 7 Defining the Learning Task (IC, quantiles), 8 Financial Feature Engineering | 6, 16 |
| 2 | Linear return forecast | 11 The ML Pipeline (ridge/lasso/logistic) | 7 |
| 3 | Pairs / stat-arb | 9 Model-Based Feature Extraction (stationarity, Kalman) | 16 |
| 4 | Volatility timing | 9 Model-Based Feature Extraction (GARCH) | 19 |
| 5 | Bayesian Sharpe / pairs | 9 (Kalman / rolling hedge ratio), 16 (Sharpe inference) | (PyMC: external) |
| 6 | Gradient-boosted long-short | 12 Gradient Boosting and Advanced Tabular Models | 6 (walk-forward CV) |
| 7 | Unsupervised risk factors | 14 Latent Factor Models (PCA eigenportfolios), 17 Portfolio Construction (HRP) | 19 |
| 8 | Text / sentiment signal | 10 Text Feature Engineering | 22 |
| 9 | Deep learning forecast | 13 Deep Learning for Time Series | |
| 10 | Autoencoder conditional factors | 14 Latent Factor Models (conditional autoencoders) | |
| 11 | Synthetic data (stretch) | 5 Synthetic Financial Data (TimeGAN) | |
| 12 | Deep RL agent (stretch) | 21 Reinforcement Learning for Execution and Hedging | |

Skills map: `data-loading` → ch. 2; `features-alpha` → ch. 7–8; `models` → ch. 6–7, 11–12;
`backtesting` → ch. 6, 16, 18; `evaluation` → ch. 7, 16; `portfolio-risk` → ch. 14, 17, 19;
`robustness` → ch. 16 (overfitting tests).

Code for strategies #1–#7: `src/quantskills/strategies/` (`factor_long_short`, `linear_forecast`,
`pairs_coint`, `vol_timing`, `bayes_pairs`, `gbm_long_short`, `unsupervised_risk`).
Strategy #5 uses an exact conjugate posterior for the Bayesian Sharpe instead of PyMC
(same answer for the normal model; PyMC stays optional for richer models).

## Papers
- Bailey, D. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *J. of Risk*. (PSR)
- Bailey, D. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *J. of Portfolio Management*.
- Bailey, Borwein, López de Prado & Zhu (2016). The Probability of Backtest Overfitting. (PBO, Phase 4)
- Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. (momentum)
- Ang, Hodrick, Xing & Zhang (2006). The Cross-Section of Volatility and Expected Returns. (low vol)
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (purged CV, DSR)
- López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *J. of Portfolio Management*. (HRP)
- Engle, R. & Granger, C. (1987). Co-integration and Error Correction. *Econometrica*. (#3)
- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors. *Econometrica*. (#3)
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *J. of Econometrics*. (#4)
- Moreira, A. & Muir, T. (2017). Volatility-Managed Portfolios. *J. of Finance*. (#4)
- Kalman, R. (1960). A New Approach to Linear Filtering and Prediction Problems. (#5)
- Ke, G. et al. (2017). LightGBM. *NeurIPS*; Lundberg, S. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). (#6)
- Markowitz, H. (1952). Portfolio Selection. *J. of Finance*; Ledoit, O. & Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. (portfolio-risk)
- Avellaneda, M. & Lee, J.-H. (2010). Statistical Arbitrage in the US Equities Market. *Quantitative Finance*. (eigenportfolios, #7)

## Libraries (licenses checked before adding)
| Library | Use | License |
|---|---|---|
| numpy, pandas, scipy | core | BSD-3 |
| pyarrow | parquet cache | Apache-2.0 |
| yfinance | free price data | Apache-2.0 |
| matplotlib | figures | PSF-based (matplotlib license) |
| textual, textual-plotext, plotext | terminal UI | MIT |
| scikit-learn | ridge / lasso / logistic, Ledoit-Wolf | BSD-3 |
| statsmodels (+ patsy) | Engle-Granger, Johansen | BSD-3 (patsy BSD-2) |
| arch | GARCH(1,1) | NCSA (permissive, BSD-like) |
| lightgbm | gradient boosting, TreeSHAP | MIT |
| pytest, ruff, mypy, nbconvert | dev | MIT / MIT / MIT / BSD-3 |

Not installed (deliberately): PyPortfolioOpt (MIT; pulls in cvxpy, so mean-variance
and HRP are written with scipy instead), shap (MIT; LightGBM computes TreeSHAP itself),
pymc (Apache-2.0; conjugate posterior suffices for now), quantstats (Apache-2.0),
vectorbt (Apache-2.0 + Commons Clause, so check it before use).

## Inspiration only
- awesome-quant (wilsonfreitas): the library landscape
- Qlib, vectorbt, backtrader, Riskfolio-Lib, PyPortfolioOpt, alphalens-reloaded: design ideas
- Fincept Terminal (AGPL-3.0, trademarked name): terminal UX inspiration only. No code, no branding.
