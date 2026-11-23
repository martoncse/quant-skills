"""Strategy library. One module per strategy; register new ones in ``STRATEGIES``."""

from typing import Any

from quantskills.strategies.base import Strategy
from quantskills.strategies.factor_long_short import FactorLongShort
from quantskills.strategies.runner import StrategyRun, run_strategy, sweep_params

STRATEGIES: dict[str, type[Strategy]] = {
    FactorLongShort.name: FactorLongShort,
}


def get_strategy(name: str, **params: Any) -> Strategy:
    try:
        return STRATEGIES[name](**params)
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None


__all__ = [
    "STRATEGIES",
    "FactorLongShort",
    "Strategy",
    "StrategyRun",
    "get_strategy",
    "run_strategy",
    "sweep_params",
]
