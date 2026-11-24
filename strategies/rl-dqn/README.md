# Strategy #12 (stretch): Deep reinforcement-learning trader (Double DQN)

**Code:** `src/quantskills/strategies/rl_agent.py`, `models/rl.py` · **Book:** ML4T 3rd ed., ch. 21 (reinforcement learning) · **Status:** Phase 5 stretch, synthetic data only · **Needs:** `uv sync --extra dl` (PyTorch + gymnasium)

## Hypothesis
Instead of predicting returns and then turning predictions into trades, let an agent learn
the trading decision directly. It is rewarded for profit net of costs, so it should learn
both when to be in the market and when trading isn't worth the cost.

## Method
1. **Environment** (`models/rl.py`, a `gymnasium.Env`):
   - **State:** the asset's return over the last 1, 5, 20 and 60 days, each in units of
     its expected size given trailing 63-day volatility, plus the position held.
     Past data only.
   - **Actions:** short, flat or long, traded at the close and held for `hold_days` (5).
   - **Reward:** position × return over the holding period, minus 10 bps per unit traded.
     These are the backtester's timing and costs.
   - **Episodes:** random 252-day stretches of the training data.
2. **Agent:** Double DQN written from scratch (replay buffer, target network, Huber loss,
   epsilon-greedy exploration decaying from 1.0 to 0.05). 20,000 steps, seeded.
3. **Validation:** the agent trains on the first 80% of the training window. Every 2,000
   steps its greedy policy is scored on the last 20%, and the best checkpoint is kept.
4. **Trading:** the frozen greedy policy runs day by day (its own position feeds back
   into the state). The default asset is SPY if present, otherwise the first ticker
   (the GARCH index `IDX` on the synthetic-plus panel).

## Results (seed 0, 10 bps costs, last 30% held out)

Illustrations of the pipeline on simulated prices, not evidence about markets.

| Panel | State | Decide every | Best validation Sharpe | In-sample Sharpe | Hold-out Sharpe | Turnover/yr |
|---|---|---|---|---|---|---|
| positive control | trend | 1 day | 0.71 | 6.56 | -1.64 | 152x |
| positive control | **trend (default)** | **5 days** | 0.24 | 3.13 | **1.82** | 28x |
| positive control | raw (last 10 returns) | 1 day | -1.46 | 9.46 | -1.28 | 216x |
| positive control | raw | 5 days | -0.71 | 3.88 | 0.31 | 44x |
| synthetic-plus `IDX` | trend | 1 day | 0.87 | 2.91 | -0.62 | 143x |
| synthetic-plus `IDX` | **trend (default)** | **5 days** | 2.68 | 1.30 | **-0.37** | 38x |
| synthetic-plus `IDX` | raw | 1 day | -0.04 | 4.24 | -2.17 | 234x |
| synthetic-plus `IDX` | raw | 5 days | 2.03 | 2.65 | -0.45 | 42x |

Benchmarks on the same hold-outs:

| Panel | Buy and hold | 20-day momentum sign (gross) | Oracle (knows the regime) |
|---|---|---|---|
| positive control | 5.02 | 4.20 | 4.43 |
| synthetic-plus `IDX` | 2.83 | 1.19 | n/a |

Default on `IDX`: 90% Bayesian interval for the hold-out Sharpe [-1.72, 0.96]; on the
control, [0.46, 3.15].

**Positive control:** `trend_regime_prices(seed=0)` is one asset whose daily drift of ±0.3%
flips sign about once every 100 days, so trend-following pays.

### What this shows
- **Daily decisions memorise.** Every daily variant scores 2.9–9.5 in-sample and loses
  on the hold-out, trading 140–230x a year. Each day's reward is mostly noise next to the
  cost, and the network learns the training days rather than a rule.
- **Weekly decisions and a compact state help.** On the positive control the default agent
  earns 1.82 out of sample, with its position agreeing with the true regime 75% of the time.
  It has learned something real.
- **But a one-line rule does better.** On the same hold-out, 20-day momentum makes 4.2.
  This hold-out happens to be mostly an up-regime, so even buy-and-hold makes 5.0. On the
  GARCH index the agent loses (-0.37) where buy-and-hold makes 2.83.
- **Validation is a weak guide** with one asset: the best validation Sharpe (2.68 on
  `IDX`) said nothing about the hold-out (-0.37).

The defaults (compact `trend` state, weekly decisions) were chosen from the 2 × 2 grid
above. That is 4 trials, chosen partly after seeing hold-out results on an earlier
version of the control panel, so treat the default rows as mildly optimistic.

## Robustness
Verdict: **RED** (`uv run python skills/robustness/scripts/robustness_report.py --strategy rl-dqn --noise-seeds 1`, about 15 minutes: every check retrains the agent). Every check runs walk-forward inside the training window except the last row.

| Check | Value | Rule | Status |
|---|---|---|---|
| Deflated Sharpe | 0.360 (SR 0.39 vs 0.66, N=4) | ≥ 0.95 green, ≥ 0.5 amber | red |
| PBO (CSCV) | 0.46, P(loss) 0.23, 3 configs | PBO ≤ 0.25 or P(loss) ≤ 0.1 green | amber |
| Sensitivity (worst / base) | — (base SR 0.39 < 0.5) | ≥ 0.5 green, ≥ 0.0 amber | n/a |
| Price noise 25 bps | — (base SR 0.39 < 0.5) | ≥ 0.75 green, ≥ 0.4 amber | n/a |
| Trade 1 day late | — (base SR 0.39 < 0.5) | ≥ 0.75 green, ≥ 0.4 amber | n/a |
| Worst regime Sharpe | 0.06 (up trend) | ≥ 0.0 green, ≥ -1.0 amber | green |
| Walk-forward / in-sample | 0.30 (SR 1.30 → 0.39) | ≥ 0.5 green, ≥ 0.0 amber | amber |
| Hold-out / in-sample | -0.28 (SR 1.30 → -0.37) | ≥ 0.5 green, ≥ 0.0 amber | red |

Walk-forward inside the training window already shows the problem: 1.30 in-sample becomes 0.39 when each block is traded by an agent trained only on earlier days, and the deflated Sharpe is red. The decision-cadence grid (walk-forward Sharpe -0.13 / 0.39 / 1.12 for 1 / 5 / 10 days) says slower is better; adopting 10 days would be a new trial.

## Failure modes
- **Memorisation:** a network that sees each training day thousands of times learns the
  days, not the pattern. Watch the gap between in-sample and validation.
- **Overtrading:** with daily decisions, the reward is mostly noise (a ±1% day is 10× the
  cost), so the agent flips constantly.
- **Reward ≠ objective:** summed P&L is not Sharpe; risk is ignored unless you put it in
  the reward.
- **Non-stationarity:** the agent learns the training regime. With one asset and five
  years there is very little experience to learn from.

## Next steps
- A risk-adjusted reward (e.g. differential Sharpe) and a position-change penalty.
- Train on many assets or many synthetic histories (#11) to give the agent more
  experience than one path through history.
