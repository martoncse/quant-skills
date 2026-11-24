"""Robustness: is a backtest's edge real, or an artefact of how it was found?

* ``pbo``: probability of backtest overfitting (CSCV)
* ``perturb``: price noise, execution delay, parameter sensitivity surfaces
* ``calibration``: Brier score, log loss, ECE, reliability tables
* ``regime``: past-only volatility / trend regimes and per-regime metrics
* ``text_attack``: adversarial stress for news-driven signals
* ``report``: all of the above plus DSR, graded red / amber / green
"""

from quantskills.robustness.calibration import (
    brier_decomposition,
    brier_score,
    calibration_summary,
    expected_calibration_error,
    log_loss,
    reliability_table,
)
from quantskills.robustness.pbo import PBOResult, pbo_cscv
from quantskills.robustness.perturb import (
    add_price_noise,
    delay_test,
    delay_weights,
    insample_backtest,
    noise_test,
    sensitivity_surface,
)
from quantskills.robustness.plots import (
    plot_decay,
    plot_generators,
    plot_pbo,
    plot_reliability,
    plot_stress,
    plot_surface,
)
from quantskills.robustness.regime import (
    REGIME_KINDS,
    market_returns,
    regime_labels,
    regime_report,
    regime_table,
)
from quantskills.robustness.report import SENSITIVITY_AXES, RobustnessReport, robustness_report
from quantskills.robustness.text_attack import ATTACKS, attack_news, text_attack_test
from quantskills.robustness.verdict import AMBER, GREEN, NA, RED, THRESHOLDS, Check, grade, overall

__all__ = [
    "AMBER",
    "ATTACKS",
    "GREEN",
    "NA",
    "RED",
    "REGIME_KINDS",
    "SENSITIVITY_AXES",
    "THRESHOLDS",
    "Check",
    "PBOResult",
    "RobustnessReport",
    "add_price_noise",
    "attack_news",
    "brier_decomposition",
    "brier_score",
    "calibration_summary",
    "delay_test",
    "delay_weights",
    "expected_calibration_error",
    "grade",
    "insample_backtest",
    "log_loss",
    "market_returns",
    "noise_test",
    "overall",
    "pbo_cscv",
    "plot_decay",
    "plot_generators",
    "plot_pbo",
    "plot_reliability",
    "plot_stress",
    "plot_surface",
    "regime_labels",
    "regime_report",
    "regime_table",
    "reliability_table",
    "robustness_report",
    "sensitivity_surface",
    "text_attack_test",
]
