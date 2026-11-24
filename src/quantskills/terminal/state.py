"""Terminal session state: which data is loaded and the last backtest run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantskills.data import attach_news, load_cached, synthetic_news, synthetic_prices
from quantskills.strategies import StrategyRun

EMPTY_CACHE_MESSAGE = (
    "The price cache is empty. Fetch data with\n"
    "  uv run python skills/data-loading/scripts/fetch.py --tickers SPY QQQ IWM\n"
    "or type  DATA DEMO  to explore with synthetic prices."
)


@dataclass
class TerminalState:
    cache_dir: Path = Path("data")
    prices: pd.DataFrame | None = None
    source: str = "none"  # "cache" | "demo" | "none" | "custom"
    message: str = ""
    last_run: StrategyRun | None = None
    last_error: str = ""
    theme: str = "qs-dark"

    def load(self, source: str) -> None:
        """Load ``"cache"`` (offline parquet cache) or ``"demo"`` (synthetic)."""
        source = source.lower()
        if source == "demo":
            prices = synthetic_prices(n_assets=30, n_days=1260, seed=0, n_pairs=3, garch_index=True)
            # Synthetic headlines ride along for the text strategy (#8).
            self.prices = attach_news(prices, synthetic_news(prices, seed=0))
            self.source = "demo"
            self.message = (
                "Synthetic demo prices and headlines: illustrations only, not market data."
            )
        elif source == "cache":
            try:
                self.prices = load_cached(self.cache_dir)
                self.source = "cache"
                self.message = f"Cached prices from {self.cache_dir}/"
            except FileNotFoundError:
                self.prices = None
                self.source = "none"
                self.message = EMPTY_CACHE_MESSAGE
        else:
            raise ValueError(f"unknown data source {source!r}")
        self.last_run = None

    @property
    def tickers(self) -> list[str]:
        if self.prices is None:
            return []
        return sorted(self.prices.index.get_level_values("ticker").unique())
