# quant-skills

A library of tested **Claude skills** for quantitative finance and ML for trading.
Each skill is a self-contained unit covering one stage of the research pipeline:
data, features, models, backtesting, portfolio risk, evaluation and (soon) robustness.
Seven strategy families from *Machine Learning for Trading* are reimplemented
from scratch and run through the same cost-aware backtester and tear sheet.

> Everything here is research code. Backtests use historical or synthetic data
> and are **not** forecasts of real returns.

## Quick start

```bash
uv sync
uv run pytest                                                  # offline, about 30 s
uv run python skills/backtesting/scripts/run_backtest.py       # strategy #1 on synthetic data
uv run python skills/evaluation/scripts/select_and_deflate.py  # sweep → deflated Sharpe → hold-out
uv run python skills/evaluation/scripts/strategy_gallery.py    # all 7 strategies, one table
uv run python skills/data-loading/scripts/fetch.py --tickers SPY QQQ IWM  # real data into data/
uv run qs-terminal --demo                                      # the terminal, on synthetic data
```

## QS Terminal

A keyboard-driven research terminal (Textual): watchlist, quotes, charts,
a strategy lab that backtests any of the seven strategies in a background worker,
a results view with in-sample vs hold-out metrics, equity, drawdown and the trade
list, and a risk screen (`RISK hrp`, F4) with weights, VaR/CVaR and a clustered
correlation heatmap. Type `HELP` inside it, or see [docs/terminal.md](docs/terminal.md).

![QS Terminal watchlist](docs/img/terminal-watchlist.png)
![QS Terminal backtest results](docs/img/terminal-results.png)
![QS Terminal risk screen](docs/img/terminal-risk.png)

## What's different
- **Costs on by default** (5 + 5 bps). Weights drift between rebalances and turnover is charged on the drifted book.
- **Look-ahead tests:** `assert_no_lookahead` scrambles the future and checks
  that past feature values don't move. Every factor and all seven strategies are tested with it,
  and a second test checks that `fit` gives the same answer when the hold-out is scrambled.
- **Purged, embargoed CV** for every ML model. The models demo shows shuffled K-fold
  "finding" an IC of 0.29 on pure noise.
- **Honest reporting:** every tear sheet carries PSR and the **deflated Sharpe
  ratio** over all trials tried, and the hold-out is kept separate.
- **Null tests:** the synthetic generator can plant a known signal or none at
  all, so we can check that a method finds the first and not the second.

## Results gallery

All numbers are on **synthetic** prices: illustrations of the machinery, not
evidence about markets. Default params, 10 bps costs, last 30% held out and touched once.
Synthetic-plus panel = 30 one-factor stocks + 3 planted cointegrated pairs + a GARCH index
(`skills/evaluation/scripts/strategy_gallery.py`).

| # | Strategy | Book ch. | In-sample Sharpe | Hold-out Sharpe | Hold-out 90% CI (Bayes) | Write-up |
|---|---|---|---|---|---|---|
| 1 | Factor long-short | 7, 8 | 0.99 | 0.10 | [-1.25, 1.42] | [README](strategies/factor-long-short/README.md) |
| 2 | Linear forecast (ridge, purged CV) | 11 | 1.23 | -0.07 | [-1.42, 1.26] | [README](strategies/linear-forecast/README.md) |
| 3 | Pairs, cointegration | 9 | 1.87 | 2.56 | [1.20, 3.90] | [README](strategies/pairs-coint/README.md) |
| 4 | Volatility timing (GARCH) | 9 | 0.61 | 2.87 * | [1.51, 4.21] | [README](strategies/vol-timing/README.md) |
| 5 | Bayesian pairs (Kalman) | 9, 16 | 2.87 | 1.10 | [-0.37, 2.55] | [README](strategies/bayes-pairs/README.md) |
| 6 | LightGBM long-short + SHAP | 12 | 3.47 | 0.21 | [-1.14, 1.54] | [README](strategies/gbm-long-short/README.md) |
| 7 | Unsupervised risk (HRP) | 14, 17 | 0.56 | -0.60 | [-1.95, 0.73] | [README](strategies/unsupervised-risk/README.md) |

\* Buy-and-hold of the same index scores 2.83: the vol target controls risk (10.3% vs 16.6% vol), not return.

What the table teaches:
- **Flexible models overfit in-sample.** LightGBM goes from 3.47 to 0.21 on the hold-out.
  Purged CV kept it to 3 trees, and it still overfit.
- **Only planted structure survives the hold-out.** The pairs strategies find the three planted
  pairs. Without a multiple-testing correction, one fluke pair turns the pairs hold-out negative.
- **A single Sharpe hides its uncertainty.** Most hold-out intervals include zero.

![pairs equity](strategies/pairs-coint/figures/synthetic_plus_equity.png)

Strategy #1 on the plain synthetic panel, and its noise-only control, is in its
[write-up](strategies/factor-long-short/README.md): the in-sample Sharpe is 1.42
with a signal and 1.32 without one. After a 12-config sweep, the DSR is 0.56.

## Skills

| Skill | Status | What it does |
|---|---|---|
| [data-loading](skills/data-loading/SKILL.md) | ✅ | yfinance + parquet cache, validation, synthetic panels |
| [features-alpha](skills/features-alpha/SKILL.md) | ✅ | momentum / reversal / low-vol factors, labels, IC |
| [backtesting](skills/backtesting/SKILL.md) | ✅ | cost-aware backtester, `Strategy` interface, train/hold-out runner |
| [evaluation](skills/evaluation/SKILL.md) | ✅ | metrics, tear sheet, PSR / DSR, Bayesian Sharpe, parameter sweeps |
| [models](skills/models/SKILL.md) | ✅ | feature/label panels, purged + embargoed CV, ridge/lasso/logistic, LightGBM + SHAP |
| [portfolio-risk](skills/portfolio-risk/SKILL.md) | ✅ | Ledoit-Wolf, min-variance, mean-variance, HRP, PCA eigenportfolios, VaR/CVaR |
| robustness | ⏳ Phase 4 | PBO, perturbation tests, calibration |
| fundamental-pitch | ⏳ Phase 6 | DCF + one-page pitch |

## Build checklist

- [x] **Phase 1: Foundation:** scaffold, CI, data-loading, backtesting, evaluation, `Strategy` interface, strategy #1
- [x] **Phase 2: Terminal MVP:** command bar, watchlist, quote, chart, strategy lab, results (`uv run qs-terminal`)
- [x] **Phase 3: Core strategies:** #2–#7, models, portfolio-risk, risk screen (`RISK`, F4)
- [ ] Phase 4: Robustness module + screen + write-up
- [ ] Phase 5: Strategies #8–#10 (`dl` extra), stretch #11–#12
- [ ] Phase 6: fundamental-pitch, terminal polish

## Layout
Logic lives in `src/quantskills/`. Skills hold `SKILL.md`, thin scripts, a
demo notebook (under 2 min) and tests. See [CLAUDE.md](CLAUDE.md) for conventions
and [docs/references.md](docs/references.md) for sources and the book chapter map.

License: MIT.
