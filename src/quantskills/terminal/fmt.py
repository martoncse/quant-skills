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


def _blend(a: str, b: str, t: float) -> str:
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


HEAT_BASE, HEAT_POS, HEAT_NEG = "#161b22", "#1f6feb", "#da3633"


def heat(value: float, text: str | None = None) -> Text:
    """A heatmap cell for a value in [-1, 1]: blue for positive, red for negative."""
    if _bad(value):
        return Text(text or "—", style=MUTED)
    t = min(abs(float(value)), 1.0) ** 0.6  # gamma: make modest correlations visible
    bg = _blend(HEAT_BASE, HEAT_POS if value >= 0 else HEAT_NEG, t)
    fg = "#ffffff" if t > 0.45 else "#c9d1d9"
    return Text(text if text is not None else f"{value:+.2f}", style=f"{fg} on {bg}")
