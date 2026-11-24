"""Performance metrics, factor IC analysis, deflated Sharpe and tear sheets."""

from quantskills.eval.bayes import BayesianSharpe, bayesian_sharpe
from quantskills.eval.ic import ic_summary, information_coefficient, quantile_returns
from quantskills.eval.metrics import (
    TRADING_DAYS,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summary,
)
from quantskills.eval.sharpe import (
    DeflatedSharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from quantskills.eval.tearsheet import format_tear_sheet, plot_tear_sheet, tear_sheet

__all__ = [
    "TRADING_DAYS",
    "BayesianSharpe",
    "bayesian_sharpe",
    "DeflatedSharpe",
    "annualized_return",
    "annualized_volatility",
    "calmar_ratio",
    "deflated_sharpe_ratio",
    "drawdown_series",
    "expected_max_sharpe",
    "format_tear_sheet",
    "ic_summary",
    "information_coefficient",
    "max_drawdown",
    "plot_tear_sheet",
    "probabilistic_sharpe_ratio",
    "quantile_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "summary",
    "tear_sheet",
]
