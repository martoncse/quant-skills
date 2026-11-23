# QS Terminal

A keyboard-driven research terminal built with [Textual](https://textual.textualize.io/).
It is a thin UI over `quantskills`: it parses commands, calls library
functions and draws the results. It holds no quant logic of its own.

```bash
uv run qs-terminal            # cached prices from data/ (offline)
uv run qs-terminal --demo     # synthetic prices, no download needed
```

If the cache is empty, the watchlist says so and tells you how to fill it
(`skills/data-loading/scripts/fetch.py`) or to switch to `DATA DEMO`.

## Commands

| Command | What it does |
|---|---|
| `WATCH` | Watchlist: last, daily change, 52-week range, 3M vol, 60-day sparkline |
| `QUOTE <TICKER>` or just `<TICKER>` | Quote detail with 1M / 3M / YTD / 1Y returns |
| `CHART <TICKER> [1M\|3M\|6M\|1Y\|2Y\|5Y\|MAX]` | Close with SMA 50 / 200 |
| `STRAT [strategy]` | Strategy lab: edit params, hold-out fraction and costs, then run |
| `BT <strategy> [key=value ...]` | Run a backtest right away, e.g. `BT factor-long-short quantile=0.1` |
| `RESULTS` | Re-open the last backtest's results |
| `DATA DEMO\|CACHE` | Switch data source |
| `RISK`, `ROBUST` | Placeholders for Phases 3 and 4 |
| `HELP`, `QUIT` | |

Keys: `F1` help, `F2` watchlist, `F3` lab, `ctrl+r` run (in the lab), `Esc` back,
`/` focus the command bar, `ctrl+q` quit.

## Screens

- **Watchlist**: `quote_table()` from `quantskills.data.views`. Enter on a row opens the quote.
- **Quote**: `quote_detail()`.
- **Chart**: `sma()` from `quantskills.features.indicators`, computed on full
  history and then windowed, so the averages are valid from the first bar shown.
- **Strategy lab**: the param form is generated from `Strategy.default_params()`.
  The backtest (`run_strategy`) runs in a Textual worker thread, so the UI stays responsive.
- **Results**: in-sample vs hold-out tear sheets (`tear_sheet`), equity with the
  hold-out boundary marked, drawdown, and a trade list from `BacktestResult.trades()`
  (trades are measured from the drifted book, so they match the costs charged).

## Layout

```
src/quantskills/terminal/
├── app.py        # App, command dispatch, CSS, entry point
├── commands.py   # command grammar (pure, unit-tested)
├── screens.py    # one Screen class per view
├── state.py      # loaded data + last run
└── fmt.py        # display formatting only
```

## Tests

`tests/terminal/test_app.py` drives each screen with Textual's `App.run_test()` pilot:
empty-cache message, watchlist → quote → back, chart and help, the lab running a
backtest in a worker, `BT` with good and bad params, and invalid lab input.
`tests/terminal/test_commands.py` covers the parser.

Original code and look. Fincept Terminal and Bloomberg were UX inspiration only;
no code or branding is used.
