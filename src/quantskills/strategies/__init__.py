"""Strategy library. One module per strategy; register new ones in ``STRATEGIES``."""

from typing import Any

from quantskills.strategies.base import Strategy
from quantskills.strategies.bayes_pairs import BayesianPairs
from quantskills.strategies.cond_autoencoder import ConditionalAutoencoder
from quantskills.strategies.dl_forecast import DLForecast
from quantskills.strategies.factor_long_short import FactorLongShort
from quantskills.strategies.gbm_long_short import GBMLongShort
from quantskills.strategies.linear_forecast import LinearForecast
from quantskills.strategies.pairs_coint import PairsCointegration
from quantskills.strategies.runner import StrategyRun, run_strategy, sweep_params
from quantskills.strategies.text_sentiment import TextSentiment
from quantskills.strategies.unsupervised_risk import UnsupervisedRisk
from quantskills.strategies.vol_timing import VolatilityTiming

# Book order: #1 factor, #2 linear, #3 pairs, #4 vol timing, #5 Bayesian pairs,
# #6 gradient boosting, #7 unsupervised risk, #8 text sentiment, #9 deep learning,
# #10 conditional autoencoder. #9 and #10 need the optional `dl` extra (PyTorch),
# imported only when they are fitted, so the registry always lists them.
STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (
        FactorLongShort,
        LinearForecast,
        PairsCointegration,
        VolatilityTiming,
        BayesianPairs,
        GBMLongShort,
        UnsupervisedRisk,
        TextSentiment,
        DLForecast,
        ConditionalAutoencoder,
    )
}


def get_strategy(name: str, **params: Any) -> Strategy:
    try:
        return STRATEGIES[name](**params)
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None


__all__ = [
    "STRATEGIES",
    "BayesianPairs",
    "ConditionalAutoencoder",
    "DLForecast",
    "FactorLongShort",
    "GBMLongShort",
    "LinearForecast",
    "PairsCointegration",
    "Strategy",
    "StrategyRun",
    "TextSentiment",
    "UnsupervisedRisk",
    "VolatilityTiming",
    "get_strategy",
    "run_strategy",
    "sweep_params",
]
