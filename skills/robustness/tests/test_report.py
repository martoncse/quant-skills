import pandas as pd
import pytest

from quantskills.data import synthetic_prices
from quantskills.robustness import GREEN, RED, THRESHOLDS, Check, grade, overall, robustness_report
from quantskills.robustness.verdict import AMBER, NA, retention
from quantskills.strategies import FactorLongShort
from quantskills.utils import holdout_date

FAST = {
    "axes": {"quantile": (0.1, 0.2), "rebalance_days": (10, 21)},
    "noise_bps": (25.0,),
    "noise_seeds": (0,),
    "delays": (0, 1),
    "n_blocks": 8,
}


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_prices(n_assets=20, n_days=900, seed=4)


@pytest.fixture(scope="module")
def train_end(prices) -> pd.Timestamp:
    return holdout_date(prices.index.get_level_values("date"), 0.3)


@pytest.fixture(scope="module")
def report(prices, train_end):
    return robustness_report("factor-long-short", prices, train_end, **FAST)


def test_report_has_every_check(report):
    names = [c.name for c in report.checks]
    for expected in (
        "Deflated Sharpe",
        "PBO (CSCV)",
        "Sensitivity (worst / base)",
        "Price noise 25 bps",
        "Trade 1 day late",
        "Worst regime Sharpe",
        "Walk-forward / in-sample",
        "Hold-out / in-sample",
    ):
        assert expected in names
    assert report.verdict in {GREEN, AMBER, RED}
    assert report.dsr.n_trials == 1 + 4  # prior trials + the grid
    assert report.pbo is not None and report.pbo.n_trials == 4
    assert report.surface.shape == (2, 2)
    table = report.checks_table()
    assert list(table.columns) == ["check", "value", "rule", "status"]
    assert "verdict" in report.to_markdown()
    assert report.noise_summary().index[0] == 0.0


def test_only_the_holdout_line_depends_on_the_holdout(prices, train_end):
    """Scramble the hold-out: every training-window check must be unchanged."""
    later = prices.index.get_level_values("date") > train_end
    scrambled = prices.copy()
    for col in ("open", "high", "low", "close"):
        scrambled.loc[later, col] = scrambled.loc[later, col].to_numpy()[::-1]
    a = robustness_report("factor-long-short", prices, train_end, **FAST)
    b = robustness_report("factor-long-short", scrambled, train_end, **FAST)
    for ca, cb in zip(a.checks, b.checks, strict=True):
        if ca.name == "Hold-out / in-sample":
            continue
        assert ca.display == cb.display, ca.name
    assert a.is_sharpe == b.is_sharpe and a.wf_sharpe == b.wf_sharpe


def test_in_sample_mode_and_instance_input(prices, train_end):
    r = robustness_report(
        FactorLongShort(quantile=0.1), prices, train_end, mode="in-sample", **FAST
    )
    assert r.params["quantile"] == 0.1
    assert r.mode == "in-sample" and r.base_sharpe == r.is_sharpe
    assert "Walk-forward / in-sample" not in [c.name for c in r.checks]


def test_no_grid_means_no_pbo(prices, train_end):
    r = robustness_report(
        "factor-long-short", prices, train_end, **{**FAST, "axes": {}}, mode="in-sample"
    )
    pbo = next(c for c in r.checks if c.name == "PBO (CSCV)")
    assert pbo.status == NA and r.pbo is None


def test_grading_rules():
    assert grade(0.96, *THRESHOLDS["dsr"]) == GREEN
    assert grade(0.6, *THRESHOLDS["dsr"]) == AMBER
    assert grade(0.1, *THRESHOLDS["dsr"]) == RED
    assert grade(float("nan"), 0, 0) == RED
    assert grade(0.2, *THRESHOLDS["pbo"], higher_is_better=False) == GREEN
    assert grade(0.7, *THRESHOLDS["pbo"], higher_is_better=False) == RED
    assert retention(1.0, 2.0) == 0.5
    assert pd.isna(retention(1.0, -0.5))  # nothing to retain from a losing base
    checks = [Check("a", 0, "", "", GREEN), Check("b", 0, "", "", AMBER), Check("c", 0, "", "", NA)]
    assert overall(checks) == AMBER
    assert overall([Check("c", 0, "", "", NA)]) == NA
