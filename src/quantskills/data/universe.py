"""Example universes, with a survivorship-bias warning."""

from __future__ import annotations

import warnings

# Large US stocks that exist *today*. Back-testing on them is survivorship biased:
# names that were delisted, acquired or shrank out of the index are missing.
EXAMPLE_LARGE_CAPS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "XOM", "UNH", "HD", "MA", "CVX", "KO", "PEP", "MRK", "ABBV", "WMT",
    "BAC", "COST", "DIS", "CSCO", "INTC", "VZ", "T", "PFE", "ORCL", "MCD",
]  # fmt: skip


class SurvivorshipBiasWarning(UserWarning):
    """Universe built from current constituents."""


def survivorship_warning(universe_name: str = "EXAMPLE_LARGE_CAPS") -> None:
    """Emit a warning that ``universe_name`` was built from current constituents."""
    warnings.warn(
        f"{universe_name} is built from current constituents: backtests on it are "
        "survivorship-biased and will look better than reality.",
        SurvivorshipBiasWarning,
        stacklevel=2,
    )
