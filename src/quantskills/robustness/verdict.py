"""Red / amber / green grading. Thresholds live here, in one place, so they are easy to audit.

A verdict is a **triage flag**, not proof. Green means "no obvious sign of
overfitting or fragility in these particular tests", never "this will make money".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GREEN, AMBER, RED, NA = "green", "amber", "red", "n/a"
_ORDER = {GREEN: 0, AMBER: 1, RED: 2}


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    display: str  # formatted value
    rule: str  # plain-language threshold
    status: str  # green / amber / red / n/a
    note: str = ""


def grade(value: float, green: float, amber: float, higher_is_better: bool = True) -> str:
    """Green if past ``green``, amber if past ``amber``, else red. NaN -> red."""
    if value is None or math.isnan(value):
        return RED
    if higher_is_better:
        return GREEN if value >= green else AMBER if value >= amber else RED
    return GREEN if value <= green else AMBER if value <= amber else RED


def retention(value: float, base: float) -> float:
    """``value / base`` when the base Sharpe is positive; NaN otherwise (nothing to retain)."""
    if base is None or math.isnan(base) or base <= 0 or math.isnan(value):
        return float("nan")
    return value / base


def overall(checks: list[Check]) -> str:
    """Worst status across graded checks."""
    graded = [c.status for c in checks if c.status in _ORDER]
    if not graded:
        return NA
    return max(graded, key=_ORDER.__getitem__)


# Retention ratios (perturbed / base Sharpe) are only graded above this base Sharpe.
MIN_BASE_SHARPE = 0.5

# (green, amber) thresholds; see docs/robustness.md for the reasoning.
THRESHOLDS: dict[str, tuple[float, float]] = {
    "dsr": (0.95, 0.50),  # P(true SR > best-of-N noise), in-sample
    "pbo": (0.25, 0.50),  # lower is better
    "pbo_loss": (0.10, 0.30),  # P(IS winner has OOS Sharpe < 0), lower is better
    "noise": (0.75, 0.40),  # Sharpe kept after 25 bps price noise
    "delay": (0.75, 0.40),  # Sharpe kept when trading 1 day late
    "surface": (0.50, 0.00),  # worst grid Sharpe / base Sharpe
    "holdout": (0.50, 0.00),  # hold-out Sharpe / in-sample Sharpe
    "regime": (0.00, -1.00),  # worst regime Sharpe (losing in one regime is common)
}
