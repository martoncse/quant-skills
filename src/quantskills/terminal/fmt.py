"""Display formatting only (strings and colours). No calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rich.text import Text

BLOCKS = "▁▂▃▄▅▆▇█"

# One palette per terminal theme (the Textual themes are registered in app.py from these).
PALETTES: dict[str, dict[str, str]] = {
    "qs-dark": {
        "background": "#0b0f14", "surface": "#11161d", "panel": "#161b22", "border": "#30363d",
        "foreground": "#c9d1d9", "muted": "#8b949e", "accent": "#d29922", "line": "#39c5bb",
        "up": "#3fb950", "down": "#f85149", "warn": "#d29922",
        "heat_base": "#161b22", "heat_pos": "#1f6feb", "heat_neg": "#da3633",
    },
    "qs-amber": {  # a nod to monochrome phosphor terminals
        "background": "#0a0700", "surface": "#120d00", "panel": "#1a1300", "border": "#4d3600",
        "foreground": "#ffb000", "muted": "#a87400", "accent": "#ffd27f", "line": "#ffcc4d",
        "up": "#ffe08a", "down": "#ff6a3d", "warn": "#ffb000",
        "heat_base": "#1a1300", "heat_pos": "#ffb000", "heat_neg": "#ff4d1a",
    },
    "qs-light": {
        "background": "#ffffff", "surface": "#f6f8fa", "panel": "#eaeef2", "border": "#d0d7de",
        "foreground": "#1f2328", "muted": "#59636e", "accent": "#9a6700", "line": "#0969da",
        "up": "#1a7f37", "down": "#cf222e", "warn": "#9a6700",
        "heat_base": "#ffffff", "heat_pos": "#0969da", "heat_neg": "#cf222e",
    },
}  # fmt: skip
THEMES = tuple(PALETTES)
_current = {"name": "qs-dark"}


def set_palette(name: str) -> None:
    if name not in PALETTES:
        raise ValueError(f"unknown theme {name!r}; choose from {', '.join(THEMES)}")
    _current["name"] = name


def palette_name() -> str:
    return _current["name"]


def c(key: str) -> str:
    """A colour from the current palette, e.g. ``c("up")``."""
    return PALETTES[_current["name"]][key]


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


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
        return Text(text, style=c("muted"))
    return Text(text, style=c("up") if value > 0 else c("down"))


def _blend(a: str, b: str, t: float) -> str:
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


def heat(value: float, text: str | None = None) -> Text:
    """A heatmap cell for a value in [-1, 1]: blue for positive, red for negative."""
    if _bad(value):
        return Text(text or "—", style=c("muted"))
    t = min(abs(float(value)), 1.0) ** 0.6  # gamma: make modest correlations visible
    bg = _blend(c("heat_base"), c("heat_pos") if value >= 0 else c("heat_neg"), t)
    fg = "#ffffff" if _luminance(bg) < 0.5 else "#1f2328"
    return Text(text if text is not None else f"{value:+.2f}", style=f"{fg} on {bg}")


STATUS_KEY = {"green": "up", "amber": "warn", "red": "down"}


def status_text(status: str, text: str | None = None) -> Text:
    """A red / amber / green verdict label (colours follow the theme)."""
    key = STATUS_KEY.get(status)
    style = f"bold {c(key)}" if key else c("muted")
    return Text(text if text is not None else status.upper(), style=style)
