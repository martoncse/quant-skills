"""Pilot tests: one basic flow per screen. Synthetic data only, no network."""

import asyncio

import pytest
from textual.widgets import DataTable, Input, OptionList

from quantskills.data import attach_news, synthetic_news, synthetic_prices
from quantskills.strategies import STRATEGIES
from quantskills.terminal.app import QSTerminal
from quantskills.terminal.screens import (
    ChartScreen,
    HelpScreen,
    QuoteScreen,
    ResultsScreen,
    RiskScreen,
    RobustScreen,
    StrategyLabScreen,
    WatchlistScreen,
)

SIZE = (120, 40)


@pytest.fixture(scope="module")
def small_prices():
    return synthetic_prices(n_assets=12, n_days=600, seed=3)


@pytest.fixture(scope="module")
def rich_prices():
    """Enough structure for every strategy: pairs to find, a GARCH index to time."""
    prices = synthetic_prices(n_assets=15, n_days=700, seed=3, n_pairs=2, garch_index=True)
    return attach_news(prices, synthetic_news(prices, seed=3, p_news=0.2))


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
            # 30 synthetic stocks + 3 cointegrated pairs + a GARCH index
            assert app.screen.query_one("#watch", DataTable).row_count == 37

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


def test_risk_screen_shows_weights_var_and_heatmap(rich_prices):
    async def flow():
        app = QSTerminal(prices=rich_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "RISK")
            assert isinstance(app.screen, RiskScreen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            assert screen.query_one("#risk-weights", DataTable).row_count == 20
            assert screen.query_one("#risk-metrics", DataTable).row_count >= 8
            corr = screen.query_one("#risk-corr", DataTable)
            assert corr.row_count == 12 and len(corr.columns) == 13
            metrics = screen.query_one("#risk-metrics", DataTable)
            assert metrics.get_row_at(2)[0] == "VaR 95% (hist)"
            await pilot.press("escape")
            await _type(pilot, "RISK min_var 126")
            await app.workers.wait_for_complete()
            assert app.screen.method == "min_var" and app.screen.lookback == 126
            await _type(pilot, "RISK magic")
            assert "unknown method" in app.state.last_error
            await _type(pilot, "RISK hrp 5")
            assert "lookback" in app.state.last_error

    run(flow())


def test_risk_screen_reports_bad_window(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "RISK hrp 5000")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # 600 days of data: the window is clipped to what exists, not an error
            assert app.screen.query_one("#risk-weights", DataTable).row_count == 12

    run(flow())


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_runs_from_the_terminal(name, rich_prices):
    fast = {
        "gbm-long-short": " max_rounds=30",
        "unsupervised-risk": " lookback=126",
        "dl-forecast": " max_epochs=2 lookback=63",
        "cond-autoencoder": " max_epochs=20",
        "rl-dqn": " train_steps=400 eval_every=200",
    }
    if name in {"dl-forecast", "cond-autoencoder", "rl-dqn"}:
        pytest.importorskip("torch")
    if name == "rl-dqn":
        pytest.importorskip("gymnasium")

    async def flow():
        app = QSTerminal(prices=rich_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, f"BT {name}{fast.get(name, '')}")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.state.last_error == ""
            assert isinstance(app.screen, ResultsScreen), name
            assert app.state.last_run.strategy.name == name

    run(flow())


def test_robust_screen_shows_verdict_checks_and_surface(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "ROBUST factor-long-short")
            assert isinstance(app.screen, RobustScreen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            screen = app.screen
            checks = screen.query_one("#robust-checks", DataTable)
            names = [str(checks.get_row_at(i)[0]) for i in range(checks.row_count)]
            assert names[0] == "Deflated Sharpe" and names[-1] == "Hold-out / in-sample"
            assert "PBO (CSCV)" in names
            statuses = {str(checks.get_row_at(i)[1]) for i in range(checks.row_count)}
            assert statuses <= {"GREEN", "AMBER", "RED", "N/A"}
            verdict = str(screen.query_one("#robust-verdict").render())
            assert "Verdict" in verdict
            surface = screen.query_one("#robust-surface", DataTable)
            assert surface.row_count == 3 and len(surface.columns) == 4
            assert screen.query_one("#robust-perturb", DataTable).row_count == 4 + 4
            assert screen.query_one("#robust-regimes", DataTable).row_count >= 2

    run(flow())


def test_robust_defaults_to_last_run_and_validates(small_prices):
    async def flow():
        app = QSTerminal(prices=small_prices)
        async with app.run_test(size=SIZE) as pilot:
            await _type(pilot, "BT factor-long-short quantile=0.1")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("f5")
            await pilot.pause()
            assert isinstance(app.screen, RobustScreen)
            assert app.screen.strategy.params["quantile"] == 0.1
            await app.workers.wait_for_complete()
            await pilot.press("escape")
            await _type(pilot, "ROBUST nope")
            assert "Unknown strategy" in app.state.last_error
            await _type(pilot, "ROBUST factor-long-short quantile=0.9")
            assert "quantile" in app.state.last_error

    run(flow())
