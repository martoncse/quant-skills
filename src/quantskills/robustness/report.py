"""One call that runs every robustness check on a strategy and grades it.

Discipline: everything below runs on the **training window** except one line,
the hold-out Sharpe, which comes from the single ``run_strategy`` call. So the
report can be re-run freely without spending the hold-out, *as long as you do
not re-tune after reading that line*.

Inside the training window the checks are scored **walk-forward** by default:
the model that trades each day was fitted only on earlier days. Scoring on the
days a model was fitted to hides exactly the overfitting we are looking for
(LightGBM passes every in-sample check on synthetic data, then fails the hold-out).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantskills.backtest.engine import CostModel
from quantskills.data.loader import to_wide
from quantskills.eval.metrics import sharpe_ratio
from quantskills.eval.sharpe import DeflatedSharpe, deflated_sharpe_ratio
from quantskills.robustness.pbo import PBOResult, pbo_cscv
from quantskills.robustness.perturb import (
    delay_test,
    noise_test,
    sensitivity_surface,
    train_backtest,
)
from quantskills.robustness.regime import regime_report
from quantskills.robustness.text_attack import text_attack_test
from quantskills.robustness.verdict import (
    AMBER,
    GREEN,
    MIN_BASE_SHARPE,
    NA,
    RED,
    THRESHOLDS,
    Check,
    grade,
    overall,
    retention,
)
from quantskills.strategies import STRATEGIES, Strategy, run_strategy

# Two parameters per strategy, three values each, centred on the defaults.
SENSITIVITY_AXES: dict[str, dict[str, tuple[Any, ...]]] = {
    "factor-long-short": {"quantile": (0.1, 0.2, 0.3), "rebalance_days": (10, 21, 42)},
    "linear-forecast": {"quantile": (0.1, 0.2, 0.3), "rebalance_days": (10, 21, 42)},
    "pairs-coint": {"entry_z": (1.5, 2.0, 2.5), "z_window": (42, 63, 126)},
    "vol-timing": {"target_vol": (0.05, 0.10, 0.15), "rebalance_days": (1, 5, 21)},
    "bayes-pairs": {"entry_z": (1.5, 2.0, 2.5), "delta": (1e-6, 1e-5, 1e-4)},
    "gbm-long-short": {"num_leaves": (7, 15, 31), "min_data_in_leaf": (100, 200, 400)},
    "unsupervised-risk": {"lookback": (126, 252, 504), "rebalance_days": (10, 21, 42)},
    "text-sentiment": {"window_days": (3, 5, 10), "quantile": (0.2, 0.25, 0.33)},
    "dl-forecast": {"hidden": (8, 16, 32), "dropout": (0.0, 0.1, 0.3)},
    "cond-autoencoder": {"n_factors": (1, 3, 5), "hidden": (8, 16, 32)},
}


@dataclass
class RobustnessReport:
    strategy: str
    params: dict[str, Any]
    train_end: pd.Timestamp
    is_sharpe: float
    wf_sharpe: float  # walk-forward inside the training window (nan in "in-sample" mode)
    oos_sharpe: float
    dsr: DeflatedSharpe
    pbo: PBOResult | None
    surface: pd.DataFrame | None
    noise: pd.DataFrame
    delay: pd.DataFrame
    regimes: pd.DataFrame
    text_attacks: pd.DataFrame | None = None  # news strategies only
    mode: str = "walk-forward"
    checks: list[Check] = field(default_factory=list)

    @property
    def base_sharpe(self) -> float:
        """Sharpe the perturbation checks are measured against."""
        return self.wf_sharpe if self.mode == "walk-forward" else self.is_sharpe

    @property
    def verdict(self) -> str:
        return overall(self.checks)

    def checks_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"check": c.name, "value": c.display, "rule": c.rule, "status": c.status}
                for c in self.checks
            ]
        )

    def noise_summary(self) -> pd.DataFrame:
        """Mean in-sample Sharpe per noise level (0 bps = clean)."""
        s = self.noise.groupby("bps")["sharpe"].mean()
        s = pd.concat([pd.Series({0.0: self.base_sharpe}), s])
        s.index.name = "bps"
        return s.rename("sharpe").to_frame()

    def to_markdown(self) -> str:
        lines = [
            f"**{self.strategy}**: verdict **{self.verdict.upper()}** "
            f"(train <= {self.train_end.date()})",
            "",
            "| Check | Value | Rule | Status |",
            "|---|---|---|---|",
        ]
        for c in self.checks:
            lines.append(f"| {c.name} | {c.display} | {c.rule} | {c.status} |")
        return "\n".join(lines)


def _fmt(x: float, spec: str = "{:.2f}") -> str:
    return "—" if x is None or np.isnan(x) else spec.format(x)


def robustness_report(
    strategy: str | Strategy,
    data: pd.DataFrame,
    train_end: str | pd.Timestamp,
    costs: CostModel | None = None,
    axes: Mapping[str, Sequence[Any]] | None = None,
    noise_bps: Sequence[float] = (10.0, 25.0, 50.0),
    noise_seeds: Sequence[int] = (0, 1),
    delays: Sequence[int] = (0, 1, 2, 5),
    n_blocks: int = 16,
    n_trials: int = 1,
    mode: str = "walk-forward",
    n_folds: int = 4,
    progress: Callable[[str], None] | None = None,
) -> RobustnessReport:
    """Run DSR, PBO, noise, delay, sensitivity, regime and hold-out checks.

    ``strategy`` is a registry name or an (unfitted) instance whose ``params``
    are the configuration under test. ``axes`` overrides the sensitivity grid
    (two parameters); by default :data:`SENSITIVITY_AXES` is used. ``n_trials``
    counts configurations you tried *before* this report; the grid's own trials
    are added on top for the deflated Sharpe. ``mode="in-sample"`` scores the
    checks on the days each model was fitted to (faster, and over-optimistic).
    """
    say = progress or (lambda _msg: None)
    if isinstance(strategy, str):
        strategy = STRATEGIES[strategy]()
    cls, params = type(strategy), dict(strategy.params)
    ts = pd.Timestamp(train_end)
    train = data[data.index.get_level_values("date") <= ts]
    close = to_wide(train, "close")

    say("base run (fit on train, one look at the hold-out)")
    run = run_strategy(cls(**params), data, ts, costs)
    is_ret = run.in_sample.trim_warmup().returns
    is_sr = sharpe_ratio(is_ret)
    oos_sr = sharpe_ratio(run.out_of_sample.trim_warmup().returns)  # as in tear_sheet

    wf_sr = float("nan")
    if mode == "walk-forward":
        say(f"walk-forward base ({n_folds} folds inside the training window)")
    base_res, base_w = train_backtest(cls, params, train, costs, mode, n_folds)
    base_ret = base_res.returns
    if mode == "walk-forward":
        wf_sr = sharpe_ratio(base_ret)

    surface, pbo = None, None
    trial_sharpes: list[float] = []
    axes = axes if axes is not None else SENSITIVITY_AXES.get(cls.name)
    if axes:
        say(f"sensitivity grid over {', '.join(axes)}")
        surface, trials = sensitivity_surface(cls, params, axes, train, costs, mode, n_folds)
        trial_sharpes = [float(x) for x in surface.to_numpy().ravel() if not np.isnan(x)]
        if trials.shape[1] >= 2 and len(trials) >= 4 * n_blocks:
            say(f"PBO over {trials.shape[1]} configs, {n_blocks} blocks")
            pbo = pbo_cscv(trials, n_blocks)
    total_trials = max(1, n_trials) + len(trial_sharpes)
    dsr = deflated_sharpe_ratio(
        base_ret, n_trials=total_trials, trial_sharpes=trial_sharpes or None
    )

    say("price-noise test")
    noise = noise_test(cls, params, train, noise_bps, noise_seeds, costs, mode, n_folds)
    say("execution-delay test")
    delay = delay_test(base_w, close, delays, costs)
    say("regimes")
    regimes = regime_report(base_ret, close)
    text_attacks = None
    if getattr(cls, "uses_news", False):
        say("adversarial text attacks")
        text_attacks = text_attack_test(cls, params, train, costs=costs)

    base_sr = wf_sr if mode == "walk-forward" else is_sr
    checks = _grade(
        base_sr, is_sr, wf_sr, oos_sr, dsr, pbo, surface, noise, delay, regimes, noise_bps
    )
    if text_attacks is not None:
        checks.insert(-2 if mode == "walk-forward" else -1, _text_check(text_attacks))
    return RobustnessReport(
        strategy=cls.name,
        params=params,
        train_end=ts,
        is_sharpe=is_sr,
        wf_sharpe=wf_sr,
        oos_sharpe=oos_sr,
        dsr=dsr,
        pbo=pbo,
        surface=surface,
        noise=noise,
        delay=delay,
        regimes=regimes,
        text_attacks=text_attacks,
        mode=mode,
        checks=checks,
    )


def _text_check(t: pd.DataFrame) -> Check:
    T = THRESHOLDS["text"]
    rule = f"≥ {T[0]} green, ≥ {T[1]} amber"
    base = float(t.loc[t["attack"] == "none", "sharpe"].iloc[0])
    hit = t[t["attack"] != "none"]
    worst = hit.loc[hit["sharpe"].idxmin()]
    if np.isnan(base) or base < MIN_BASE_SHARPE:
        return Check(
            "Adversarial text",
            np.nan,
            f"— (base SR {_fmt(base)} < {MIN_BASE_SHARPE})",
            rule,
            NA,
            note="base Sharpe too small",
        )
    r = float(worst["sharpe"]) / base
    return Check(
        "Adversarial text",
        r,
        f"{_fmt(r)} kept (worst: {worst['attack']}, SR {_fmt(float(worst['sharpe']))})",
        rule,
        grade(r, *T),
    )


def _grade(
    base_sr: float,
    is_sr: float,
    wf_sr: float,
    oos_sr: float,
    dsr: DeflatedSharpe,
    pbo: PBOResult | None,
    surface: pd.DataFrame | None,
    noise: pd.DataFrame,
    delay: pd.DataFrame,
    regimes: pd.DataFrame,
    noise_bps: Sequence[float],
) -> list[Check]:
    T = THRESHOLDS
    checks = [
        Check(
            "Deflated Sharpe",
            dsr.dsr,
            f"{dsr.dsr:.3f} (SR {dsr.sharpe:.2f} vs {dsr.sharpe_threshold:.2f}, N={dsr.n_trials})",
            f"≥ {T['dsr'][0]} green, ≥ {T['dsr'][1]} amber",
            grade(dsr.dsr, *T["dsr"]),
        )
    ]
    if pbo is not None:
        # A high PBO on a plateau (every config profitable) is harmless: picking the
        # IS winner is a coin flip, but the coin pays either way. So grade on the
        # better of PBO and P(IS winner loses OOS).
        by_pbo = grade(pbo.pbo, *T["pbo"], higher_is_better=False)
        by_loss = grade(pbo.prob_oos_loss, *T["pbo_loss"], higher_is_better=False)
        status = min(by_pbo, by_loss, key=[GREEN, AMBER, RED].index)
        checks.append(
            Check(
                "PBO (CSCV)",
                pbo.pbo,
                f"{pbo.pbo:.2f}, P(loss) {pbo.prob_oos_loss:.2f}, {pbo.n_trials} configs",
                f"PBO ≤ {T['pbo'][0]} or P(loss) ≤ {T['pbo_loss'][0]} green",
                status,
            )
        )
    else:
        checks.append(Check("PBO (CSCV)", np.nan, "—", "needs a parameter grid", NA))

    def kept(name: str, value: float, base: float, key: str, detail: str) -> Check:
        rule = f"≥ {T[key][0]} green, ≥ {T[key][1]} amber"
        if np.isnan(base) or base < MIN_BASE_SHARPE:
            # A ratio against a near-zero Sharpe is noise; weak strategies are
            # already flagged by the deflated Sharpe.
            msg = f"— (base SR {_fmt(base)} < {MIN_BASE_SHARPE})"
            return Check(name, np.nan, msg, rule, NA, note="base Sharpe too small")
        r = value / base
        return Check(name, r, f"{_fmt(r)} kept ({detail})", rule, grade(r, *T[key]))

    if surface is not None:
        worst = float(np.nanmin(surface.to_numpy())) if surface.notna().any().any() else np.nan
        checks.append(
            kept("Sensitivity (worst / base)", worst, base_sr, "surface", f"worst SR {_fmt(worst)}")
        )

    level = 25.0 if 25.0 in noise_bps else float(np.median(noise_bps))
    noisy_sr = float(noise.loc[noise["bps"] == level, "sharpe"].mean())
    checks.append(
        kept(f"Price noise {level:g} bps", noisy_sr, base_sr, "noise", f"SR {_fmt(noisy_sr)}")
    )

    d = delay.set_index("delay")["sharpe"]
    if 1 in d.index:
        late = float(d[1])
        checks.append(
            kept("Trade 1 day late", late, float(d.get(0, base_sr)), "delay", f"SR {_fmt(late)}")
        )

    worst_regime = float(regimes["sharpe"].min()) if len(regimes) else np.nan
    where = regimes["sharpe"].idxmin() if regimes["sharpe"].notna().any() else None
    checks.append(
        Check(
            "Worst regime Sharpe",
            worst_regime,
            f"{_fmt(worst_regime)}" + (f" ({where[1]})" if where else ""),
            f"≥ {T['regime'][0]} green, ≥ {T['regime'][1]} amber",
            grade(worst_regime, *T["regime"]),
        )
    )

    if not np.isnan(wf_sr):
        r = retention(wf_sr, is_sr)
        checks.append(
            Check(
                "Walk-forward / in-sample",
                r,
                f"{_fmt(r)} (SR {_fmt(is_sr)} → {_fmt(wf_sr)})",
                f"≥ {T['holdout'][0]} green, ≥ {T['holdout'][1]} amber",
                grade(r, *T["holdout"]),
                note="training window only: refit-on-all vs fit-on-past",
            )
        )
    r = retention(oos_sr, is_sr)
    checks.append(
        Check(
            "Hold-out / in-sample",
            r,
            f"{_fmt(r)} (SR {_fmt(is_sr)} → {_fmt(oos_sr)})",
            f"≥ {T['holdout'][0]} green, ≥ {T['holdout'][1]} amber",
            grade(r, *T["holdout"]),
            note="the only line that uses the hold-out",
        )
    )
    return checks
