import pytest

from quantskills.terminal.commands import Command, CommandError, parse_command
from quantskills.terminal.fmt import sparkline


@pytest.mark.parametrize(
    "text, expected",
    [
        ("help", Command("HELP")),
        ("?", Command("HELP")),
        ("quote aapl", Command("QUOTE", ("AAPL",))),
        ("AAPL", Command("QUOTE", ("AAPL",))),
        ("brk.b", Command("QUOTE", ("BRK.B",))),
        ("chart spy", Command("CHART", ("SPY", "1Y"))),
        ("CHART SPY 5y", Command("CHART", ("SPY", "5Y"))),
        ("strat", Command("STRAT")),
        ("STRAT Factor-Long-Short", Command("STRAT", ("factor-long-short",))),
        ("data demo", Command("DATA", ("DEMO",))),
        ("w", Command("WATCH")),
        ("risk", Command("RISK", ("hrp", "252"))),
        ("RISK MIN_VAR 126", Command("RISK", ("min_var", "126"))),
    ],
)
def test_parse_ok(text, expected):
    assert parse_command(text) == expected


def test_parse_bt_params():
    cmd = parse_command("BT factor-long-short quantile=0.1 rebalance_days=5 weighting=equal")
    assert cmd.name == "BT"
    assert cmd.args == ("factor-long-short",)
    assert cmd.params == {"quantile": 0.1, "rebalance_days": 5, "weighting": "equal"}


@pytest.mark.parametrize(
    "text",
    ["", "   ", "chart", "chart spy 7Y", "quote a b", "bt", "bt x quantile", "data web",
     "help me", "foo bar", "risk bogus", "risk hrp x", "risk hrp 10",
     "risk hrp 100 extra"],
)  # fmt: skip
def test_parse_errors(text):
    with pytest.raises(CommandError):
        parse_command(text)


def test_sparkline():
    assert sparkline([1, 2, 3, 4, 5, 6, 7, 8]) == "▁▂▃▄▅▆▇█"
    assert sparkline([2, 2, 2]) == "▄▄▄"
    assert sparkline([]) == ""
    assert len(sparkline([1.0, float("nan"), 3.0])) == 2
