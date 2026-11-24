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
        ("robust", Command("ROBUST")),
        ("dcf", Command("DCF")),
        ("DCF cases/My.json", Command("DCF", ("cases/My.json",))),
        ("theme", Command("THEME")),
        ("THEME Amber", Command("THEME", ("qs-amber",))),
        ("ROBUST Pairs-Coint", Command("ROBUST", ("pairs-coint",))),
        ("robust pairs-coint entry_z=2.5", Command("ROBUST", ("pairs-coint",), {"entry_z": 2.5})),
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
     "risk hrp 100 extra", "robust pairs-coint entry_z", "theme neon", "theme dark light",
     "dcf a.json b.json"],
)  # fmt: skip
def test_parse_errors(text):
    with pytest.raises(CommandError):
        parse_command(text)


def test_sparkline():
    assert sparkline([1, 2, 3, 4, 5, 6, 7, 8]) == "▁▂▃▄▅▆▇█"
    assert sparkline([2, 2, 2]) == "▄▄▄"
    assert sparkline([]) == ""
    assert len(sparkline([1.0, float("nan"), 3.0])) == 2


def test_help_groups_cover_every_command_once():
    from quantskills.terminal.commands import COMMAND_GROUPS, COMMANDS

    grouped = [n for names in COMMAND_GROUPS.values() for n in names]
    assert sorted(grouped) == sorted(COMMANDS) and len(grouped) == len(set(grouped))


def test_palettes_drive_colours():
    from quantskills.terminal import fmt

    try:
        fmt.set_palette("qs-dark")
        dark = (fmt.c("up"), str(fmt.heat(0.8).style), str(fmt.status_text("red").style))
        fmt.set_palette("qs-light")
        light = (fmt.c("up"), str(fmt.heat(0.8).style), str(fmt.status_text("red").style))
        assert dark != light and all(a != b for a, b in zip(dark, light, strict=True))
        # readable text on every heat background: dark text on pale cells, light on strong
        assert str(fmt.heat(0.05).style).startswith("#1f2328")
        assert set(fmt.PALETTES) == set(fmt.THEMES)
        keys = {frozenset(p) for p in fmt.PALETTES.values()}
        assert len(keys) == 1  # every palette defines the same keys
        try:
            fmt.set_palette("neon")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown theme accepted")
    finally:
        fmt.set_palette("qs-dark")
