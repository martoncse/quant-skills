"""Pilot tests: one basic flow per screen. Synthetic data only, no network."""

import asyncio

import pytest
from textual.widgets import DataTable, Input, OptionList

from quantskills.data import synthetic_prices
from quantskills.terminal.app import QSTerminal
from quantskills.terminal.screens import (
    ChartScreen,
    HelpScreen,
    QuoteScreen,
    ResultsScreen,
    StrategyLabScreen,
    WatchlistScreen,
)

SIZE = (120, 40)


@pytest.fixture(scope="module")
def small_prices():
    return synthetic_prices(n_assets=12, n_days=600, seed=3)


async def _type(pilot, text: str) -> None:
    cmd = pilot.app.screen.query_one("#cmd", Input)
    cmd.focus()
    cmd.value = text
    await pilot.press("enter")
    await pilot.pause()


def run(coro):
    return asyncio.run(coro)


def test_empty_cache_shows_clear_message(tmp_path):
    async def flow():
        app = QSTerminal(source="cache", cache_dir=tmp_path)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, WatchlistScreen)
            assert app.state.prices is None
            assert "cache is empty" in app.state.message.lower()
            assert app.screen.query_one("#watch", DataTable).display is False
            await _type(pilot, "QUOTE SPY")
            assert "No data" in app.state.last_error
            await _type(pilot, "DATA DEMO")
            assert app.state.source == "demo"
            assert app.screen.query_one("#watch", DataTable).row_count == 30

    run(flow())


def test_watchlist_quote_and_back(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            table = app.screen.query_one("#watch", DataTable)
            assert table.row_count == 12
            await _type(pilot, "SYN03")
            assert isinstance(app.screen, QuoteScreen)
            assert app.screen.ticker == "SYN03"
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, WatchlistScreen)
            await _type(pilot, "QUOTE NOPE")
            assert "Unknown ticker" in app.state.last_error
            assert isinstance(app.screen, WatchlistScreen)

    run(flow())


def test_chart_and_help(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "CHART SYN01 6M")
            assert isinstance(app.screen, ChartScreen)
            assert app.screen.period == "6M"
            await _type(pilot, "HELP")
            assert isinstance(app.screen, HelpScreen)
            # usage strings contain [brackets]; they must not be eaten as Rich markup
            assert "key=value" in app.export_screenshot()
            await _type(pilot, "WATCH")
            assert isinstance(app.screen, WatchlistScreen)
            await _type(pilot, "chart spy 9Y")
            assert "unknown period" in app.state.last_error

    run(flow())


def test_strategy_lab_runs_backtest_in_worker(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "STRAT")
            assert isinstance(app.screen, StrategyLabScreen)
            lab = app.screen
            assert lab.query_one("#strategies", OptionList).option_count >= 1
            lab.query_one("#param-quantile", Input).value = "0.25"
            await pilot.press("ctrl+r")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ResultsScreen)
            assert app.state.last_run is not None
            assert app.state.last_run.strategy.params["quantile"] == 0.25
            assert app.screen.query_one("#metrics", DataTable).row_count > 5
            assert app.screen.query_one("#trades", DataTable).row_count > 0

    run(flow())


def test_bt_command_and_bad_params(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "BT factor-long-short not_a_param=1")
            assert "unknown params" in app.state.last_error
            await _type(pilot, "BT nope")
            assert "Unknown strategy" in app.state.last_error
            await _type(pilot, "RESULTS")
            assert "No backtest" in app.state.last_error
            await _type(pilot, "BT factor-long-short rebalance_days=5")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ResultsScreen)
            assert app.state.last_run.strategy.params["rebalance_days"] == 5
            await _type(pilot, "WATCH")
            await _type(pilot, "RESULTS")
            assert isinstance(app.screen, ResultsScreen)

    run(flow())


def test_lab_reports_invalid_settings(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "STRAT factor-long-short")
            app.screen.query_one("#param-quantile", Input).value = "0.9"
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert "Invalid settings" in app.state.last_error
            assert isinstance(app.screen, StrategyLabScreen)

    run(flow())
