"""Terminal screens. Display only: every number comes from ``quantskills``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
)
from textual_plotext import PlotextPlot

from quantskills.backtest import CostModel
from quantskills.data import period_window, quote_detail, quote_table, to_wide
from quantskills.eval import drawdown_series, tear_sheet
from quantskills.features import sma
from quantskills.fundamental import (
    Case,
    comps_table,
    dcf_value,
    implied_growth,
    scenario_table,
    sensitivity,
    valuation_label,
)
from quantskills.portfolio import RiskView, risk_view
from quantskills.robustness import RobustnessReport, robustness_report
from quantskills.strategies import STRATEGIES, Strategy, StrategyRun, get_strategy, run_strategy
from quantskills.terminal.commands import COMMAND_GROUPS, COMMANDS, parse_value
from quantskills.terminal.fmt import (
    c,
    heat,
    num,
    pct,
    signed,
    sparkline,
    status_text,
)
from quantskills.utils import holdout_date

if TYPE_CHECKING:
    from quantskills.terminal.app import QSTerminal


def _date_ticks(plt: Any, index: pd.Index, n: int = 5) -> None:
    if len(index) == 0:
        return
    pos = sorted(set(np.linspace(0, len(index) - 1, min(n, len(index))).astype(int)))
    plt.xticks(pos, [pd.Timestamp(index[i]).strftime("%Y-%m-%d") for i in pos])


def _label(text: str) -> Text:
    return Text(text, style=c("muted"))


class QSScreen(Screen):
    """Base screen: header, command bar, body, footer."""

    BINDINGS = [
        Binding("escape", "app.back", "Back"),
        Binding("slash", "focus_command", "Command", show=False),
    ]

    @property
    def qs(self) -> QSTerminal:
        return self.app  # type: ignore[return-value]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="› command  (HELP for the list)", id="cmd")
        yield from self.compose_body()
        yield Footer()

    def compose_body(self) -> ComposeResult:
        yield from ()

    @on(Input.Submitted, "#cmd")
    def _submit(self, event: Input.Submitted) -> None:
        text = event.value
        event.input.value = ""
        if text.strip():
            self.qs.run_command(text)

    def action_focus_command(self) -> None:
        self.query_one("#cmd", Input).focus()


WATCH_COLUMNS = [
    ("Ticker", 8), ("Last", 10), ("Chg", 9), ("Chg %", 8), ("52w Low", 10),
    ("52w High", 10), ("Vol 3M", 7), ("60d trend", 30),
]  # fmt: skip


class WatchlistScreen(QSScreen):
    def compose_body(self) -> ComposeResult:
        yield Static(id="status", classes="status")
        yield DataTable(id="watch", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        self.refresh_view()

    def on_screen_resume(self) -> None:
        self.refresh_view()

    def refresh_view(self) -> None:
        state = self.qs.state
        status = self.query_one("#status", Static)
        table = self.query_one("#watch", DataTable)
        table.clear(columns=True)
        if state.prices is None:
            status.update(Text(state.message, style=c("accent")))
            table.display = False
            return
        table.display = True
        quotes = quote_table(state.prices)
        close = to_wide(state.prices, "close")
        as_of = pd.Timestamp(quotes["as_of"].max()).date()
        status.update(
            f"[b]{state.source.upper()}[/b] · {len(quotes)} tickers · as of {as_of} · "
            f"{state.message}  (Enter on a row → quote)"
        )
        for label, width in WATCH_COLUMNS:
            table.add_column(label, width=width)
        for ticker, row in quotes.iterrows():
            hist = close[ticker].dropna().iloc[-60:]
            trend = float(hist.iloc[-1] - hist.iloc[0]) if len(hist) > 1 else 0.0
            table.add_row(
                Text(str(ticker), style="bold"),
                num(row["last"]),
                signed(num(row["change"]), row["change"]),
                signed(pct(row["change_pct"]), row["change_pct"]),
                num(row["low_52w"]),
                num(row["high_52w"]),
                "—" if pd.isna(row["vol_3m"]) else f"{row['vol_3m']:.1%}",
                Text(sparkline(hist.to_numpy()[::2]), style=c("up") if trend >= 0 else c("down")),
                key=str(ticker),
            )

    @on(DataTable.RowSelected, "#watch")
    def _open_quote(self, event: DataTable.RowSelected) -> None:
        self.qs.run_command(f"QUOTE {event.row_key.value}")


class QuoteScreen(QSScreen):
    def __init__(self, ticker: str) -> None:
        super().__init__()
        self.ticker = ticker

    def compose_body(self) -> ComposeResult:
        yield Static(id="quote-title", classes="panel")
        yield DataTable(id="quote", cursor_type="none", show_header=False)
        yield Static(id="quote-spark", classes="panel")

    def on_mount(self) -> None:
        prices = self.qs.state.prices
        assert prices is not None
        d = quote_detail(prices, self.ticker)
        as_of = pd.Timestamp(d["as_of"]).date()
        self.query_one("#quote-title", Static).update(
            Text.assemble((self.ticker, "bold " + c("accent")), (f"  ·  as of {as_of}", c("muted")))
        )
        t = self.query_one("#quote", DataTable)
        t.add_column("Field", width=16)
        t.add_column("Value", width=30)
        t.add_row(_label("Last"), num(d["last"]))
        change = f"{num(d['change'])}  ({pct(d['change_pct'])})"
        t.add_row(_label("Change"), signed(change, d["change"]))
        for period in ("1M", "3M", "YTD", "1Y"):
            v = d[f"ret_{period}"]
            t.add_row(_label(f"Return {period}"), signed(pct(v), v))
        t.add_row(_label("52w range"), f"{num(d['low_52w'])} - {num(d['high_52w'])}")
        vol = "—" if pd.isna(d["vol_3m"]) else f"{d['vol_3m']:.1%}"
        t.add_row(_label("Vol (3M, ann.)"), vol)
        first = pd.Timestamp(d["first_date"]).date()
        t.add_row(_label("History"), f"{d['n_days']} days since {first}")

        hist = to_wide(prices, "close")[self.ticker].dropna().iloc[-120:]
        self.query_one("#quote-spark", Static).update(
            Text.assemble(
                ("6M  ", c("muted")),
                (sparkline(hist.to_numpy()[::2]), c("line")),
                (f"\n\nCHART {self.ticker} 1Y  for the full chart", c("muted")),
            )
        )


class ChartScreen(QSScreen):
    def __init__(self, ticker: str, period: str = "1Y") -> None:
        super().__init__()
        self.ticker = ticker
        self.period = period

    def compose_body(self) -> ComposeResult:
        yield PlotextPlot(id="chart")

    def on_mount(self) -> None:
        prices = self.qs.state.prices
        assert prices is not None
        close = to_wide(prices, "close")[self.ticker].dropna()
        # Indicators use the full history, then we window, so the SMAs are valid
        # from the first plotted bar.
        frame = pd.DataFrame({"close": close, "sma50": sma(close, 50), "sma200": sma(close, 200)})
        w = period_window(frame, self.period)
        plt = self.query_one("#chart", PlotextPlot).plt
        plt.clear_data()
        x = list(range(len(w)))
        plt.plot(x, w["close"].tolist(), label=self.ticker, color="cyan")
        for col, label, color in (("sma50", "SMA 50", "orange"), ("sma200", "SMA 200", "magenta")):
            valid = w[col].notna().to_numpy()
            if valid.any():
                xs = [i for i, ok in zip(x, valid, strict=True) if ok]
                plt.plot(xs, w[col][valid].tolist(), label=label, color=color)
        _date_ticks(plt, w.index)
        plt.title(f"{self.ticker} · {self.period} · close with SMA 50/200")


class StrategyLabScreen(QSScreen):
    BINDINGS = [Binding("ctrl+r", "run", "Run backtest")]

    def __init__(
        self, name: str | None = None, params: dict[str, Any] | None = None, autorun: bool = False
    ) -> None:
        super().__init__()
        self.names = list(STRATEGIES)  # book order, #1 first
        self.current = name or self.names[0]
        self._overrides = dict(params or {})
        self._autorun = autorun

    def compose_body(self) -> ComposeResult:
        with Horizontal(id="lab"):
            yield OptionList(*self.names, id="strategies")
            with VerticalScroll(id="params"):
                yield Static(id="lab-title")
                with Horizontal(id="run-row"):
                    yield Button("Run backtest  (ctrl+r)", id="run", variant="primary")
                    yield Static(id="lab-status", classes="status")
                yield Vertical(id="param-fields")
                yield Label("hold-out fraction (last part of history, touched once)")
                yield Input("0.3", id="opt-test_fraction")
                yield Label("commission bps")
                yield Input("5", id="opt-commission_bps")
                yield Label("slippage bps")
                yield Input("5", id="opt-slippage_bps")

    async def on_mount(self) -> None:
        self.query_one("#strategies", OptionList).highlighted = self.names.index(self.current)
        await self.build_fields(self.current, self._overrides)
        if self._autorun:
            self._autorun = False
            self.start_run()

    async def build_fields(self, name: str, overrides: dict[str, Any]) -> None:
        cls = STRATEGIES[name]
        summary = (cls.__doc__ or "").strip().split("\n")[0]
        self.query_one("#lab-title", Static).update(
            Text.assemble((name, "bold " + c("accent")), ("\n" + summary, c("muted")))
        )
        container = self.query_one("#param-fields", Vertical)
        await container.remove_children()
        widgets: list[Label | Input] = []
        for key, default in cls.default_params().items():
            value = overrides.get(key, default)
            shown = value if isinstance(value, str) else repr(value)
            widgets += [Label(key), Input(shown, id=f"param-{key}", classes="param")]
        await container.mount_all(widgets)

    @on(OptionList.OptionHighlighted, "#strategies")
    async def _pick(self, event: OptionList.OptionHighlighted) -> None:
        name = str(event.option.prompt)
        if name != self.current:
            self.current = name
            await self.build_fields(name, {})

    @on(Button.Pressed, "#run")
    def _run_pressed(self) -> None:
        self.start_run()

    def action_run(self) -> None:
        self.start_run()

    def _read(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    def start_run(self) -> None:
        state = self.qs.state
        if state.prices is None:
            self.qs.report_error("No data loaded. Fetch data or type DATA DEMO.")
            return
        try:
            params = {
                inp.id.removeprefix("param-"): parse_value(inp.value.strip())
                for inp in self.query("Input.param").results(Input)
                if inp.id
            }
            strategy = get_strategy(self.current, **params)
            test_fraction = float(self._read("opt-test_fraction"))
            costs = CostModel(
                float(self._read("opt-commission_bps")), float(self._read("opt-slippage_bps"))
            )
        except Exception as exc:  # bad user input: report, don't crash
            self.qs.report_error(f"Invalid settings: {exc}")
            return
        self.query_one("#run", Button).disabled = True
        self.query_one("#lab-status", Static).update(f"Running {strategy!r} …")
        self._backtest(strategy, state.prices, test_fraction, costs)

    @work(thread=True, exclusive=True, group="backtest")
    def _backtest(
        self, strategy: Strategy, prices: pd.DataFrame, test_fraction: float, costs: CostModel
    ) -> None:
        try:
            train_end = holdout_date(prices.index.get_level_values("date"), test_fraction)
            run = run_strategy(strategy, prices, train_end, costs)
        except Exception as exc:
            self.app.call_from_thread(self._finished, None, str(exc))
            return
        self.app.call_from_thread(self._finished, run, "")

    def _finished(self, run: StrategyRun | None, error: str) -> None:
        self.query_one("#run", Button).disabled = False
        if run is None:
            self.query_one("#lab-status", Static).update(Text(f"Failed: {error}", style=c("down")))
            self.qs.report_error(f"Backtest failed: {error}")
            return
        self.query_one("#lab-status", Static).update("Done.")
        self.qs.show_results(run)


METRIC_ROWS = [
    ("Sharpe (net)", "sharpe", "{:.2f}"),
    ("Sharpe (gross)", "gross_sharpe", "{:.2f}"),
    ("Ann. return", "ann_return", "{:+.2%}"),
    ("Ann. vol", "ann_vol", "{:.2%}"),
    ("Max drawdown", "max_drawdown", "{:.2%}"),
    ("Hit rate", "hit_rate", "{:.1%}"),
    ("Turnover / yr", "ann_turnover", "{:.1f}x"),
    ("Cost drag / yr", "ann_cost_drag", "{:.2%}"),
    ("PSR", "psr", "{:.3f}"),
    ("DSR", "dsr", "{:.3f}"),
    ("Trials (N)", "n_trials", "{}"),
    ("Start", "start", "{}"),
    ("End", "end", "{}"),
]


def _cell(value: Any, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return fmt.format(value)


class ResultsScreen(QSScreen):
    def __init__(self, run: StrategyRun) -> None:
        super().__init__()
        self.run = run

    def compose_body(self) -> ComposeResult:
        yield Static(id="results-title", classes="status")
        with TabbedContent(id="tabs"):
            with TabPane("Summary", id="tab-summary"), Horizontal():
                yield DataTable(id="metrics", cursor_type="none")
                yield PlotextPlot(id="equity")
            with TabPane("Drawdown", id="tab-drawdown"):
                yield PlotextPlot(id="drawdown")
            with TabPane("Trades", id="tab-trades"):
                yield DataTable(id="trades", zebra_stripes=True)

    def on_mount(self) -> None:
        run = self.run
        bps = run.full.cost_model.rate * 1e4
        caption = (
            f"\ntrain ≤ {run.train_end.date()} · net of {bps:g} bps · "
            f"{self.qs.state.source} data · backtest, not a forecast"
        )
        self.query_one("#results-title", Static).update(
            Text.assemble((repr(run.strategy), "bold"), (caption, c("muted")))
        )

        sheets = [tear_sheet(run.in_sample), tear_sheet(run.out_of_sample)]
        metrics = self.query_one("#metrics", DataTable)
        metrics.add_columns("Metric", "In-sample", "Hold-out")
        for label, key, fmt in METRIC_ROWS:
            metrics.add_row(label, *(_cell(s.get(key), fmt) for s in sheets))

        idx = run.full.returns.index
        split_x = int(idx.searchsorted(run.train_end, side="right"))
        x = list(range(len(idx)))
        eq = self.query_one("#equity", PlotextPlot).plt
        eq.clear_data()
        eq.plot(x, (1 + run.full.gross_returns).cumprod().tolist(), label="gross", color="gray")
        eq.plot(x, run.full.equity.tolist(), label="net", color="cyan")
        eq.vline(split_x, "orange")
        _date_ticks(eq, idx, 4)
        eq.title("Growth of $1 (orange line = start of hold-out)")

        dd = self.query_one("#drawdown", PlotextPlot).plt
        dd.clear_data()
        dd.plot(x, drawdown_series(run.full.returns).tolist(), color="red", fillx=True)
        dd.vline(split_x, "orange")
        _date_ticks(dd, idx)
        dd.title("Drawdown (net)")

        trades = run.full.trades().iloc[::-1].head(500)
        table = self.query_one("#trades", DataTable)
        table.add_columns("Date", "Ticker", "Before", "After", "Traded")
        for row in trades.itertuples(index=False):
            table.add_row(
                str(pd.Timestamp(row.date).date()),
                row.ticker,
                f"{row.before:+.2%}",
                f"{row.after:+.2%}",
                signed(f"{row.traded:+.2%}", row.traded),
            )


RISK_ROWS = [
    ("Ann. vol", "ann_vol", "{:.2%}"),
    ("Max drawdown", "max_drawdown", "{:.2%}"),
    ("VaR 95% (hist)", "var_hist_95", "{:.2%}"),
    ("CVaR 95% (hist)", "cvar_hist_95", "{:.2%}"),
    ("VaR 95% (Gauss)", "var_gauss_95", "{:.2%}"),
    ("VaR 99% (hist)", "var_hist_99", "{:.2%}"),
    ("CVaR 99% (hist)", "cvar_hist_99", "{:.2%}"),
    ("VaR 99% (Cornish-F.)", "var_cf_99", "{:.2%}"),
    ("Days in window", "n_days", "{:.0f}"),
]


class RiskScreen(QSScreen):
    """Weights, VaR/CVaR and a correlation heatmap. Computed in a worker thread."""

    def __init__(self, method: str = "hrp", lookback: int = 252) -> None:
        super().__init__()
        self.method = method
        self.lookback = lookback

    def compose_body(self) -> ComposeResult:
        yield Static(f"Computing {self.method} weights…", id="risk-title", classes="status")
        with Horizontal(id="risk-top"):
            yield DataTable(id="risk-weights", zebra_stripes=True, cursor_type="none")
            yield DataTable(id="risk-metrics", cursor_type="none")
        yield Static(_label("Correlation (Ledoit-Wolf), top holdings in cluster order"))
        yield DataTable(id="risk-corr", cursor_type="none")

    def on_mount(self) -> None:
        prices = self.qs.state.prices
        if prices is None:
            self.query_one("#risk-title", Static).update(Text("No data loaded.", style=c("down")))
            return
        self._compute(prices, self.method, self.lookback)

    @work(thread=True, exclusive=True, group="risk")
    def _compute(self, prices: pd.DataFrame, method: str, lookback: int) -> None:
        try:
            view = risk_view(prices, method, lookback)
        except Exception as exc:
            self.app.call_from_thread(self._failed, str(exc))
            return
        self.app.call_from_thread(self._show, view)

    def _failed(self, error: str) -> None:
        self.query_one("#risk-title", Static).update(Text(f"Failed: {error}", style=c("down")))
        self.qs.report_error(f"Risk failed: {error}")

    def _show(self, v: RiskView) -> None:
        caption = (
            f"\n{v.lookback} days {v.start.date()} → {v.end.date()} · 1-day VaR as loss · "
            "historical, not a forecast"
        )
        self.query_one("#risk-title", Static).update(
            Text.assemble((f"RISK · {v.method}", "bold " + c("accent")), (caption, c("muted")))
        )
        wt = self.query_one("#risk-weights", DataTable)
        wt.add_columns("Ticker", "Weight", "Vol", "Risk %")
        for ticker, row in v.table.iterrows():
            wt.add_row(
                str(ticker),
                signed(f"{row.weight:+.1%}", row.weight),
                f"{row.ann_vol:.1%}",
                f"{row.risk_contrib:.1%}",
            )
        mt = self.query_one("#risk-metrics", DataTable)
        mt.add_columns("Metric", "Portfolio")
        for label, key, fmt in RISK_ROWS:
            mt.add_row(label, _cell(v.metrics.get(key), fmt))
        ct = self.query_one("#risk-corr", DataTable)
        ct.add_column("")
        for t in v.corr.columns:
            ct.add_column(str(t)[:7], width=7)
        for t, row in v.corr.iterrows():
            ct.add_row(Text(str(t), style=c("accent")), *(heat(x) for x in row))


class RobustScreen(QSScreen):
    """Robustness report for one strategy, computed in a worker thread (~2-40 s)."""

    def __init__(self, strategy: Strategy, test_fraction: float = 0.3) -> None:
        super().__init__()
        self.strategy = strategy
        self.test_fraction = test_fraction

    def compose_body(self) -> ComposeResult:
        yield Static(f"ROBUST · {self.strategy.name}", id="robust-title", classes="status")
        yield Static("Starting…", id="robust-verdict", classes="panel")
        with Horizontal(id="robust-top"):
            yield DataTable(id="robust-checks", cursor_type="none")
            with Vertical(id="robust-side"):
                yield Static(_label("Sensitivity surface"), id="robust-surface-label")
                yield DataTable(id="robust-surface", cursor_type="none")
                yield Static(_label("Perturbations"))
                yield DataTable(id="robust-perturb", cursor_type="none")
                yield Static(_label("Regimes (training window)"))
                yield DataTable(id="robust-regimes", cursor_type="none")

    def on_mount(self) -> None:
        prices = self.qs.state.prices
        if prices is None:
            self.query_one("#robust-verdict", Static).update(
                Text("No data loaded.", style=c("down"))
            )
            return
        self._compute(self.strategy, prices, self.test_fraction)

    def _progress(self, message: str) -> None:
        self.query_one("#robust-verdict", Static).update(Text(f"Running: {message} …", c("muted")))

    @work(thread=True, exclusive=True, group="robust")
    def _compute(self, strategy: Strategy, prices: pd.DataFrame, test_fraction: float) -> None:
        def say(msg: str) -> None:
            self.app.call_from_thread(self._progress, msg)

        try:
            train_end = holdout_date(prices.index.get_level_values("date"), test_fraction)
            report = robustness_report(strategy, prices, train_end, progress=say)
        except Exception as exc:
            self.app.call_from_thread(self._failed, str(exc))
            return
        self.app.call_from_thread(self._show, report)

    def _failed(self, error: str) -> None:
        self.query_one("#robust-verdict", Static).update(Text(f"Failed: {error}", style=c("down")))
        self.qs.report_error(f"Robustness failed: {error}")

    def _show(self, r: RobustnessReport) -> None:
        caption = (
            f"\n{r.strategy} · train ≤ {r.train_end.date()} · checks scored {r.mode} "
            "inside the training window · only the last row uses the hold-out"
        )
        self.query_one("#robust-title", Static).update(
            Text.assemble((f"ROBUST · {r.strategy}", "bold " + c("accent")), (caption, c("muted")))
        )
        self.query_one("#robust-verdict", Static).update(
            Text.assemble(
                ("Verdict: ", "bold"),
                status_text(r.verdict),
                (
                    "   a triage flag, not proof: green means no red flags in these tests",
                    c("muted"),
                ),
            )
        )
        ct = self.query_one("#robust-checks", DataTable)
        ct.add_columns("Check", "Status", "Value", "Rule")
        for chk in r.checks:
            ct.add_row(
                chk.name, status_text(chk.status), Text(chk.display), Text(chk.rule, c("muted"))
            )

        st = self.query_one("#robust-surface", DataTable)
        if r.surface is None:
            st.add_column("no grid for this strategy")
        else:
            s = r.surface
            vals = s.to_numpy(dtype=float)
            lim = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
            self.query_one("#robust-surface-label", Static).update(
                _label(f"Sensitivity ({r.mode} Sharpe): rows {s.index.name}, cols {s.columns.name}")
            )
            st.add_column("")
            for col in s.columns:
                st.add_column(f"{col:g}", width=7)
            for idx, row in s.iterrows():
                cells = [heat(v / lim if lim else 0.0, _cell(v, "{:.2f}")) for v in row]
                st.add_row(Text(f"{idx:g}", c("accent")), *cells)

        pt = self.query_one("#robust-perturb", DataTable)
        pt.add_columns("Test", "Size", "Sharpe")
        for bps, row in r.noise_summary().iterrows():
            pt.add_row("price noise", f"{bps:g} bps", _cell(row.sharpe, "{:.2f}"))
        for row in r.delay.itertuples(index=False):
            pt.add_row("trade late", f"{row.delay} d", _cell(row.sharpe, "{:.2f}"))

        rt = self.query_one("#robust-regimes", DataTable)
        rt.add_columns("Regime", "Days", "Sharpe", "Ann. ret")
        for (_kind, name), row in r.regimes.iterrows():
            rt.add_row(
                str(name),
                f"{row.days:.0f}",
                signed(_cell(row.sharpe, "{:.2f}"), row.sharpe),
                _cell(row.ann_return, "{:+.1%}"),
            )


class DCFScreen(QSScreen):
    """DCF, scenarios, sensitivity and comps for one case file. Fast, so no worker."""

    def __init__(self, case: Case) -> None:
        super().__init__()
        self.case = case

    def compose_body(self) -> ComposeResult:
        yield Static(id="dcf-title", classes="status")
        with Horizontal(id="dcf-top"):
            yield DataTable(id="dcf-metrics", cursor_type="none")
            yield DataTable(id="dcf-scenarios", cursor_type="none")
        yield Static(_label("Value per share: WACC (rows) x terminal growth (columns)"))
        yield DataTable(id="dcf-sens", cursor_type="none")
        yield Static(_label("Peers (implied value per share)"))
        yield DataTable(id="dcf-comps", cursor_type="none")

    def on_mount(self) -> None:
        case, co = self.case, self.case.company
        v = dcf_value(case)
        g = implied_growth(case)
        tag = "  [FICTIONAL EXAMPLE]" if co.fictional else ""
        self.query_one("#dcf-title", Static).update(
            Text.assemble(
                (f"DCF · {co.name} ({co.ticker}){tag}", "bold " + c("accent")),
                ("\n" + valuation_label(v.upside), c("muted")),
                (" · illustrative, not investment advice", c("muted")),
            )
        )
        mt = self.query_one("#dcf-metrics", DataTable)
        mt.add_columns("Metric", "Value")
        rows = [
            ("Price", f"{co.price:,.2f} {co.currency}"),
            ("DCF value / share", f"{v.per_share:,.2f}"),
            ("Upside", signed(f"{v.upside:+.1%}", v.upside)),
            ("WACC", f"{v.wacc:.2%}"),
            ("Terminal growth", f"{v.terminal_growth:.2%}"),
            ("Terminal value / EV", f"{v.terminal_share:.0%}"),
            ("Price implies growth", "—" if g != g else f"{g:.1%} / yr"),
            ("Enterprise value", f"{v.enterprise_value:,.0f}m"),
        ]
        for label, val in rows:
            mt.add_row(label, val)
        st = self.query_one("#dcf-scenarios", DataTable)
        st.add_columns("Scenario", "Value / share", "Upside", "WACC", "g")
        for name, r in scenario_table(case).iterrows():
            st.add_row(
                str(name),
                f"{r.per_share:,.2f}",
                signed(f"{r.upside:+.1%}", r.upside),
                f"{r.wacc:.2%}",
                f"{r.terminal_growth:.2%}",
            )
        sens = sensitivity(case)
        sv = self.query_one("#dcf-sens", DataTable)
        sv.add_column("")
        for gcol in sens.columns:
            sv.add_column(f"g {gcol:.1%}", width=9)
        for r, row in sens.iterrows():
            # colour = value vs price: blue above, red below (+-50% is full colour)
            cells = [heat((x / co.price - 1) * 2, _cell(x, "{:,.2f}")) for x in row]
            sv.add_row(Text(f"WACC {r:.2%}", c("accent")), *cells)
        ct = self.query_one("#dcf-comps", DataTable)
        ct.add_columns("Peer", "EV/EBITDA", "P/E", "Value (EV/EBITDA)", "Value (P/E)")
        for row in comps_table(case).itertuples(index=False):
            ct.add_row(
                str(row.peer),
                _cell(row.ev_ebitda, "{:.1f}x"),
                _cell(row.pe, "{:.1f}x"),
                _cell(row.value_ev_ebitda, "{:,.2f}"),
                _cell(row.value_pe, "{:,.2f}"),
            )


KEYS = [
    ("F1 / F2 / F3", "Help / Watchlist / Strategy lab"),
    ("F4 / F5 / F6", "Risk / Robustness (last backtest) / DCF example"),
    ("Ctrl+T", "Next colour theme (dark, amber, light)"),
    ("Esc", "Back"),
    ("/", "Focus the command bar"),
    ("Ctrl+R", "Run the backtest (in the lab)"),
    ("Ctrl+P", "Textual command palette"),
    ("Ctrl+Q", "Quit"),
]
EXAMPLES = [
    "AAPL                              quote for a ticker",
    "CHART SPY 5Y                      price with SMA 50 / 200",
    "BT pairs-coint entry_z=2.5        backtest with a changed parameter",
    "ROBUST gbm-long-short             is the edge real?",
    "RISK min_var 126                  minimum-variance weights, 6-month window",
    "DCF my_case.json                  value your own case file",
]


class HelpScreen(QSScreen):
    def compose_body(self) -> ComposeResult:
        with VerticalScroll(id="help-scroll"):
            yield Static(id="help", classes="panel")

    def on_mount(self) -> None:
        self.refresh_view()

    def refresh_view(self) -> None:
        state = self.qs.state
        t = Table(
            title="QS Terminal",
            caption=(
                f"data: {state.source} · theme: {state.theme.removeprefix('qs-')} · "
                "backtests and valuations are illustrations, not forecasts"
            ),
            expand=False,
            show_lines=False,
        )
        t.add_column("Command", style=c("accent"), no_wrap=True)
        t.add_column("What it does")
        for group, names in COMMAND_GROUPS.items():
            t.add_row(Text(group, style="bold " + c("line")), "")
            for n in names:
                usage, desc = COMMANDS[n]
                # Text(), not str: "[key=value ...]" would otherwise be parsed as Rich markup.
                t.add_row(Text("  " + usage), Text(desc))
        t.add_row("", "")
        t.add_row(Text("Keys", style="bold " + c("line")), "")
        for k, d in KEYS:
            t.add_row(Text("  " + k), Text(d))
        t.add_row("", "")
        t.add_row(Text("Examples", style="bold " + c("line")), "")
        for e in EXAMPLES:
            cmd, _, what = e.partition("  ")
            t.add_row(Text("  " + cmd.strip()), Text(what.strip(), style=c("muted")))
        self.query_one("#help", Static).update(t)
