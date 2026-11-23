"""Shared pytest fixtures. All data is synthetic: tests never hit the network."""

import pandas as pd
import pytest

from quantskills.data import synthetic_prices, to_wide


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return synthetic_prices(n_assets=25, n_days=900, seed=7)


@pytest.fixture(scope="session")
def close(prices: pd.DataFrame) -> pd.DataFrame:
    return to_wide(prices, "close")
