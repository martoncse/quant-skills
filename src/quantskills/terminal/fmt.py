"""Display formatting only (strings and colours). No calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rich.text import Text

BLOCKS = "▁▂▃▄▅▆▇█"
UP, DOWN, MUTED = "#3fb950", "#f85149", "#8b949e"


def sparkline(values: Sequence[float]) -> str:
    """Unicode block sparkline; NaNs are skipped."""
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return BLOCKS[3] * len(vals)
    scale = (len(BLOCKS) - 1) / (hi - lo)
    return "".join(BLOCKS[int(round((v - lo) * scale))] for v in vals)


def _bad(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def num(x: float | None, digits: int = 2) -> str:
    return "—" if _bad(x) else f"{x:,.{digits}f}"


def pct(x: float | None, digits: int = 2) -> str:
    return "—" if _bad(x) else f"{x:+.{digits}%}"


def signed(text: str, value: float | None) -> Text:
    """Colour text green/red by the sign of ``value``."""
    if value is None or _bad(value) or value == 0:
        return Text(text, style=MUTED)
    return Text(text, style=UP if value > 0 else DOWN)
