r"""Record the README demo GIF (and still screenshots) from a scripted terminal session.

uv run python docs/tools/make_demo_gif.py
uv run python docs/tools/make_demo_gif.py --chrome /path/to/chrome

Drives QS Terminal with Textual's test pilot on the synthetic demo data, saves an SVG
screenshot after each step, renders the SVGs to PNG with headless Chrome (Textual's
screenshots are SVG), and stitches them into docs/img/terminal-demo.gif with Pillow.
Needs Chrome or Chromium; nothing else beyond the core install.
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from textual.widgets import Input

from quantskills.terminal.app import QSTerminal
from quantskills.terminal.fmt import set_palette

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "img"

# (command, caption, still-image name or "")
STEPS = [
    ("WATCH", "Watchlist", ""),
    ("CHART IDX 2Y", "Chart with moving averages", ""),
    ("BT pairs-coint", "Backtest in a worker: hold-out vs in-sample", ""),
    ("RISK hrp", "Risk: HRP weights, VaR / CVaR, correlations", ""),
    ("ROBUST factor-long-short", "Robustness verdict", ""),
    ("DCF", "Valuation of a fictional company", "terminal-dcf.png"),
    ("THEME amber", "Themes: amber", "terminal-amber.png"),
    ("THEME light", "Themes: light", ""),
    ("HELP", "Help", "terminal-help.png"),
]


def find_chrome(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("CHROME"),
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome/Chromium not found: pass --chrome PATH or set CHROME")


async def record(tmp: Path, size: tuple[int, int]) -> list[tuple[Path, str, str]]:
    frames = []
    app = QSTerminal(source="demo")
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        for i, (cmd, caption, still) in enumerate(STEPS):
            inp = app.screen.query_one("#cmd", Input)
            inp.focus()
            inp.value = cmd
            await pilot.press("enter")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause(0.3)
            if cmd.startswith("THEME"):  # show the theme on the watchlist
                while len(app.screen_stack) > 2:
                    app.pop_screen()
                await pilot.pause(0.3)
                getattr(app.screen, "refresh_view", lambda: None)()
                await pilot.pause(0.3)
            svg = tmp / f"frame{i:02d}.svg"
            app.save_screenshot(str(svg))
            frames.append((svg, caption, still))
            if cmd not in ("WATCH",) and not cmd.startswith("THEME"):
                await pilot.press("escape")
                await pilot.pause()
    set_palette("qs-dark")
    return frames


def render(chrome: str, svg: Path, png: Path, window: tuple[int, int]) -> None:
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--screenshot={png}",
            f"--window-size={window[0]},{window[1]}",
            svg.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--chrome", default=None)
    p.add_argument("--width", type=int, default=1000, help="GIF width in pixels")
    p.add_argument("--ms", type=int, default=2200, help="milliseconds per frame")
    a = p.parse_args()
    chrome = find_chrome(a.chrome)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        frames = asyncio.run(record(tmp, (130, 42)))
        images = []
        for svg, caption, still in frames:
            png = svg.with_suffix(".png")
            render(chrome, svg, png, (1600, 1100))
            im = Image.open(png).convert("RGB")
            bbox = Image.eval(im, lambda v: 255 - v).getbbox()  # trim the white page margin
            im = im.crop(bbox) if bbox else im
            if still:
                im.save(IMG / still)
            h = round(im.height * a.width / im.width)
            images.append(im.resize((a.width, h), Image.LANCZOS))
            print(f"frame: {caption}")
        gif = IMG / "terminal-demo.gif"
        images[0].save(
            gif,
            save_all=True,
            append_images=images[1:],
            duration=a.ms,
            loop=0,
            optimize=True,
        )
        print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
