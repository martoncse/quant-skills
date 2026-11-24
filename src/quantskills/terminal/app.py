"""QS Terminal: a keyboard-driven research terminal (Textual).

    uv run qs-terminal            # cached prices (offline)
    uv run qs-terminal --demo     # synthetic prices

A thin UI layer: commands are parsed in ``commands.py`` and every number is
computed in ``quantskills``. Backtests run in a worker thread.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from textual.app import App
from textual.binding import Binding

from quantskills.strategies import STRATEGIES, StrategyRun, get_strategy
from quantskills.terminal.commands import Command, CommandError, parse_command
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
from quantskills.terminal.state import TerminalState

CSS = """
Screen { background: #0b0f14; }
#cmd { border: tall #d29922; background: #11161d; }
#cmd:focus { border: tall #39c5bb; }
.status { padding: 0 1; height: auto; color: #8b949e; }
.panel { padding: 1 2; height: auto; }
DataTable { height: 1fr; }
PlotextPlot { height: 1fr; }
#lab { height: 1fr; }
#strategies { width: 28; height: 1fr; border: round #30363d; }
#params { width: 1fr; padding: 0 1; }
#param-fields { height: auto; }
#params Input {
    width: 48; height: 1; border: none; padding: 0 1; background: #161b22; margin-bottom: 1;
}
#params Input:focus { background: #1f2a36; }
#run-row { height: auto; margin: 1 0; }
#run-row .status { width: 1fr; padding: 1 2; }
#metrics { width: 44; }
#quote { height: auto; width: 50; margin: 0 2; }
#risk-top { height: 1fr; }
#risk-weights { width: 1fr; }
#risk-metrics { width: 40; }
#risk-corr { height: auto; max-height: 16; }
#robust-top { height: 1fr; }
#robust-checks { width: 3fr; }
#robust-side { width: 2fr; }
#robust-side DataTable { height: auto; max-height: 12; }
"""


class QSTerminal(App):
    TITLE = "QS Terminal"
    CSS = CSS
    BINDINGS = [
        Binding("f1", "command('HELP')", "Help"),
        Binding("f2", "command('WATCH')", "Watchlist"),
        Binding("f3", "command('STRAT')", "Lab"),
        Binding("f4", "command('RISK')", "Risk"),
        Binding("f5", "command('ROBUST')", "Robust"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        source: str = "cache",
        cache_dir: str | Path = "data",
        prices: pd.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self.state = TerminalState(cache_dir=Path(cache_dir))
        self._initial_source = source
        if prices is not None:
            self.state.prices = prices
            self.state.source = "custom"
            self.state.message = "Prices supplied by caller."

    def on_mount(self) -> None:
        if self.state.prices is None:
            self.state.load(self._initial_source)
        self.sub_title = f"data: {self.state.source}"
        self.push_screen(WatchlistScreen())

    # ----- actions -------------------------------------------------------------
    def action_command(self, text: str) -> None:
        self.run_command(text)

    async def action_back(self) -> None:
        if len(self.screen_stack) > 2:
            await self.pop_screen()

    def report_error(self, message: str) -> None:
        self.state.last_error = message
        self.notify(message, title="Error", severity="error")

    def show_results(self, run: StrategyRun) -> None:
        self.state.last_run = run
        self.push_screen(ResultsScreen(run))

    # ----- command dispatch ----------------------------------------------------
    def run_command(self, text: str) -> None:
        try:
            cmd = parse_command(text)
        except CommandError as exc:
            self.report_error(str(exc))
            return
        self.state.last_error = ""
        getattr(self, f"_cmd_{cmd.name.lower()}")(cmd)

    def _home(self) -> None:
        while len(self.screen_stack) > 2:
            self.pop_screen()

    def _have_data(self) -> bool:
        if self.state.prices is None:
            self.report_error("No data loaded. Fetch data, or type DATA DEMO.")
            return False
        return True

    def _have_ticker(self, ticker: str) -> bool:
        if not self._have_data():
            return False
        if ticker not in self.state.tickers:
            shown = ", ".join(self.state.tickers[:8])
            self.report_error(f"Unknown ticker {ticker}. Loaded: {shown}…")
            return False
        return True

    def _have_strategy(self, name: str) -> bool:
        if name not in STRATEGIES:
            self.report_error(f"Unknown strategy {name!r}. Available: {', '.join(STRATEGIES)}")
            return False
        return True

    def _cmd_watch(self, cmd: Command) -> None:
        self._home()

    def _cmd_quote(self, cmd: Command) -> None:
        if self._have_ticker(cmd.args[0]):
            self.push_screen(QuoteScreen(cmd.args[0]))

    def _cmd_chart(self, cmd: Command) -> None:
        if self._have_ticker(cmd.args[0]):
            self.push_screen(ChartScreen(*cmd.args))

    def _cmd_strat(self, cmd: Command) -> None:
        name = cmd.args[0] if cmd.args else None
        if name is None or self._have_strategy(name):
            self.push_screen(StrategyLabScreen(name))

    def _cmd_bt(self, cmd: Command) -> None:
        name = cmd.args[0]
        if not (self._have_strategy(name) and self._have_data()):
            return
        try:
            get_strategy(name, **cmd.params)  # validate params before opening the lab
        except (TypeError, ValueError) as exc:
            self.report_error(str(exc))
            return
        self.push_screen(StrategyLabScreen(name, cmd.params, autorun=True))

    def _cmd_results(self, cmd: Command) -> None:
        if self.state.last_run is None:
            self.report_error("No backtest yet. Try: BT factor-long-short")
        else:
            self.push_screen(ResultsScreen(self.state.last_run))

    def _cmd_data(self, cmd: Command) -> None:
        self.state.load(cmd.args[0])
        self.sub_title = f"data: {self.state.source}"
        self._home()
        if isinstance(self.screen, WatchlistScreen):
            self.screen.refresh_view()
        if self.state.prices is None:
            self.report_error("Cache is empty. Type DATA DEMO or run the fetch script.")
        else:
            self.notify(self.state.message, title=f"Data: {self.state.source}")

    def _cmd_risk(self, cmd: Command) -> None:
        if self._have_data():
            method, lookback = cmd.args
            self.push_screen(RiskScreen(method, int(lookback)))

    def _cmd_robust(self, cmd: Command) -> None:
        if not self._have_data():
            return
        if cmd.args:
            name, params = cmd.args[0], cmd.params
            if not self._have_strategy(name):
                return
        elif self.state.last_run is not None:  # F5: the strategy you just backtested
            last = self.state.last_run.strategy
            name, params = last.name, dict(last.params)
        else:
            name, params = "factor-long-short", {}
        try:
            strategy = get_strategy(name, **params)
        except (TypeError, ValueError) as exc:
            self.report_error(str(exc))
            return
        self.push_screen(RobustScreen(strategy))

    def _cmd_help(self, cmd: Command) -> None:
        self.push_screen(HelpScreen())

    def _cmd_quit(self, cmd: Command) -> None:
        self.exit()


def main() -> None:
    p = argparse.ArgumentParser(prog="qs-terminal", description="QS Terminal")
    p.add_argument("--demo", action="store_true", help="use synthetic prices")
    p.add_argument("--cache-dir", default="data")
    a = p.parse_args()
    QSTerminal(source="demo" if a.demo else "cache", cache_dir=a.cache_dir).run()


if __name__ == "__main__":
    main()
