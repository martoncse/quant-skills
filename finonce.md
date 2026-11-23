# CLAUDE.md — quant-skills

Guidance for Claude Code when working in this repository.

## What this project is

`quant-skills` is a portfolio repo that shows working knowledge of quantitative finance and ML for trading. It is organised as a library of **Claude skills**, where each skill is a self-contained, tested unit covering one stage of the quant research pipeline.

It draws on, but does not copy, these sources:
- **machine-learning-for-trading** (Stefan Jansen, 3rd ed.): the end-to-end workflow and chapter structure
- **awesome-quant** (wilsonfreitas): the map of libraries we build on
- **Fincept Terminal**: inspiration only, for the "terminal"/dashboard layer and agent personas
- Other references: Qlib, vectorbt, backtrader, PyPortfolioOpt, Riskfolio-Lib, alphalens-reloaded, quantstats, López de Prado's *Advances in Financial ML*

The repo has three pillars:
1. **ML strategy library:** the book's strategy families, reimplemented from scratch as runnable, backtested strategies
2. **Terminal:** a keyboard-driven financial terminal (TUI) for browsing data, running strategies and reading results
3. **Robustness module:** the differentiator. It applies adversarial-ML thinking to trading signals: overfitting diagnostics, perturbation tests and calibration.

The owner is not a traditional software engineer and builds through agentic coding tools. When you make a non-obvious design choice, explain it in plain language in the PR or commit message.

## Repository layout

```
quant-skills/
├── CLAUDE.md
├── README.md                 # portfolio-facing overview + results gallery
├── pyproject.toml            # uv-managed; single package `quantskills`
├── skills/
│   ├── data-loading/
│   │   ├── SKILL.md          # when to use, inputs, outputs, steps
│   │   ├── scripts/          # thin CLI entry points calling src/
│   │   ├── demo.ipynb        # short, runs top-to-bottom in < 2 min
│   │   └── tests/
│   ├── features-alpha/
│   ├── models/
│   ├── backtesting/
│   ├── portfolio-risk/
│   ├── evaluation/
│   ├── fundamental-pitch/
│   └── robustness/           # signature module
├── src/quantskills/          # shared, importable code (skills call into this)
│   ├── data/  features/  models/  backtest/  portfolio/  eval/  robustness/
│   ├── strategies/           # one module per book strategy (see below)
│   ├── terminal/             # Textual TUI app
│   └── utils/
├── strategies/               # per-strategy write-ups + result figures
│   └── <strategy-name>/README.md
├── data/                     # gitignored; cached downloads only
├── docs/                     # write-ups, methodology notes, references.md
└── .github/workflows/ci.yml
```

Rule: logic lives in `src/quantskills/`. Skills hold instructions, thin scripts, demos and tests. Never duplicate logic across skills.

## SKILL.md format

Every skill folder has a `SKILL.md` with YAML frontmatter:

```markdown
---
name: backtesting
description: One sentence on when to use this skill (under 200 chars).
---

## When to use
## Inputs / outputs   (types, shapes, index conventions)
## Steps
## Pitfalls           (leakage, survivorship, costs...)
## References         (book chapter, papers, upstream libs)
```

Keep SKILL.md under ~150 lines. Put deeper explanation in `docs/`.

## Tech stack and commands

- Python 3.11+, managed with **uv**
- Core: pandas, numpy, scipy, scikit-learn, lightgbm, statsmodels
- Finance: yfinance (free data), vectorbt, PyPortfolioOpt, quantstats, arch (GARCH), pymc
- Terminal: textual, textual-plotext
- Optional `dl` extra: torch, gymnasium
- Quality: ruff (lint + format), pytest, mypy (lenient)

```bash
uv sync                       # install
uv run pytest                 # all tests
uv run pytest skills/backtesting
uv run qs-terminal            # launch the terminal
uv run ruff check . --fix && uv run ruff format .
uv run jupyter nbconvert --execute --to notebook --inplace skills/*/demo.ipynb
```

Run tests and ruff before declaring any task done.

## Conventions

- Time series use a `DatetimeIndex` (tz-naive, UTC dates), sorted ascending. Multi-asset data is long format, `MultiIndex(date, ticker)`.
- Returns are simple returns unless a function name says `log_`.
- Every random process takes a `seed` argument; demos fix seeds.
- Functions are pure where possible. No hidden global state, and no network calls inside `src/` except in `data/`.
- Downloaded data is cached under `data/` (parquet). Tests never hit the network: use fixtures in `tests/fixtures/` or synthetic data.
- Type hints on public functions; short docstrings with a formula or reference where relevant.

## Quant correctness rules (non-negotiable)

1. **No look-ahead.** Features at time *t* use only data available at *t*. Shift labels, not features. Add a test for every new feature.
2. **Time-aware CV only.** Use purged/embargoed k-fold or walk-forward splits. Never use random `KFold` on time series.
3. **Costs are on by default.** Backtests include commissions and slippage parameters, with defaults that aren't zero.
4. **Survivorship bias.** Flag it whenever a universe is built from current constituents.
5. **Report honestly.** Show Sharpe with the deflated Sharpe ratio and the number of trials. Never present backtests as real or expected returns.
6. **Out-of-sample discipline.** Hold out a final test period and touch it once per experiment.

## ML strategy library (from the book)

Reimplement the strategy families from *Machine Learning for Trading*. **Before starting, check the 3rd-edition repo's chapter list** (github.com/stefan-jansen/machine-learning-for-trading) and map each strategy to its chapter in `docs/references.md`. Write all code from scratch; take only the ideas from the book.

| # | Strategy | Core method | Universe / data |
|---|----------|-------------|-----------------|
| 1 | Factor long-short | Alpha factors (momentum, value, quality) + IC analysis | US equities, daily |
| 2 | Linear return forecast | Ridge/Lasso + logistic direction model | US equities, daily |
| 3 | Pairs / stat-arb | Cointegration (Engle-Granger, Johansen), z-score entry | ETF or sector pairs |
| 4 | Volatility timing | ARIMA/GARCH forecasts → position sizing | Index ETF |
| 5 | Bayesian Sharpe / pairs | PyMC posterior on Sharpe; rolling Bayesian hedge ratio | Pairs from #3 |
| 6 | Gradient-boosted long-short | LightGBM + SHAP, purged CV | US equities, daily |
| 7 | Unsupervised risk factors | PCA eigenportfolios; HRP clustering | Equity universe |
| 8 | Text / sentiment signal | Sentiment or topic features from news/filings | Free text source |
| 9 | Deep learning forecast | MLP / 1D-CNN / LSTM on returns | Index or equities |
| 10 | Autoencoder conditional factors | Conditional autoencoder for returns | Equity panel |
| 11 | Synthetic data (stretch) | TimeGAN-style generator for stress-testing | Any |
| 12 | Deep RL agent (stretch) | DQN/PPO trading agent in a gym-style env | Single asset |

Each strategy:
- is a class implementing `Strategy` (`fit(data) -> self`, `signals(data) -> pd.DataFrame`, `params: dict`) in `src/quantskills/strategies/`
- runs through the shared backtester and evaluation tear sheet, with costs on
- gets a robustness report (deflated Sharpe, PBO, perturbation test)
- has `strategies/<name>/README.md` covering hypothesis, method, results table, failure modes and book chapter
- can be selected and run from the terminal

Strategies 9–12 need PyTorch. Add it as an optional extra (`uv sync --extra dl`) and never make it a core dependency.

## Terminal

A keyboard-driven financial terminal built with **Textual** (Python TUI), launched with `uv run qs-terminal`. It's inspired by Bloomberg/Fincept-style terminals, but every line is original code with its own name and look.

Screens (each a Textual `Screen`):
- **Command bar:** function codes, e.g. `QUOTE AAPL`, `CHART SPY 1Y`, `STRAT pairs-coint`, `BT <strategy>`, `RISK`, `ROBUST <strategy>`, `HELP`
- **Watchlist / quotes:** cached prices, daily change, sparkline
- **Chart:** price + indicators in the terminal (`textual-plotext`)
- **Strategy lab:** pick a strategy, edit params, run a backtest in a worker thread
- **Results:** equity curve, drawdown, key metrics, trade list
- **Risk:** portfolio weights, VaR/CVaR, correlation heatmap
- **Robustness:** deflated Sharpe, PBO, sensitivity surface, with a red/amber/green verdict

Rules:
- The terminal is a thin UI layer. It calls into `src/quantskills/` and holds no quant logic.
- Long jobs run in Textual workers so the UI never freezes.
- It works offline from cached data, and shows a clear message when the cache is empty.
- Test it with Textual's `App.run_test()` pilot for each screen's basic flow.

## Robustness module (priority)

`skills/robustness/` should eventually cover:
- Deflated Sharpe ratio and probability of backtest overfitting (CSCV/PBO)
- Perturbation tests: price/feature noise, small timing shifts, parameter sensitivity surfaces
- Adversarial stress on narrative/sentiment-based risk signals
- Calibration of probabilistic forecasts (Brier score, reliability diagrams, ECE)
- Regime-shift tests: how metrics degrade out of distribution

## Licensing and IP

- Write **original implementations**. Do not copy code, notebook text or prose from the ML4T book or repo. Paraphrase concepts and cite the chapter in References.
- **Fincept Terminal is AGPL-3.0 and its name is trademarked.** Copy no code and use no branding. Reference it as inspiration only.
- awesome-quant entries may be linked in `docs/references.md`. Check each dependency's license before adding it.
- Repo license: MIT, unless a dependency forces otherwise. Ask before changing it.

## Build phases

1. **Foundation:** scaffold, CI, `data-loading`, `backtesting`, `evaluation`, the `Strategy` interface, strategy #1 (factor long-short)
2. **Terminal MVP:** command bar, watchlist, chart, strategy lab and results screens, running strategy #1
3. **Core strategies:** #2–#7, plus `portfolio-risk` (mean-variance, HRP, VaR/CVaR) and the risk screen
4. **Robustness:** the module above, the robustness screen, and a `docs/` write-up with figures
5. **Advanced strategies:** #8–#10 (DL extra), then the stretch goals #11–#12
6. **Extras:** `fundamental-pitch` (DCF plus a one-page pitch), plus terminal polish (themes, help screen, demo GIF for the README)

Finish one phase before starting the next. Update the checklist in README.md as skills land.

## Definition of done (per skill)

- [ ] SKILL.md complete, in the format above
- [ ] Logic in `src/quantskills/`, with type hints
- [ ] Tests pass offline, including a look-ahead test where relevant
- [ ] `demo.ipynb` executes cleanly in under 2 minutes
- [ ] README results gallery updated with one figure or table

## Working style for Claude

- Before a multi-file change, state a short plan.
- Prefer small, reviewable commits with clear messages.
- If a request would break a correctness rule above, say so and propose the correct approach. Don't do it silently.
- Don't add heavy dependencies (e.g. PyTorch, zipline) without asking.
