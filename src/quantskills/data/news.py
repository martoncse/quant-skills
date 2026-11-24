"""Synthetic headlines (offline, reproducible) and how news travels with prices.

News is a DataFrame with columns ``date, ticker, headline``: a headline dated *t*
is published during day *t* and may be acted on at the close of *t*.

It travels with a price panel in ``prices.attrs["news"]`` (see :func:`attach_news`),
so every ``Strategy`` keeps the plain ``fit(data)`` / ``signals(data)`` interface and
slicing the prices by date carries the news along. Strategies must still cut the
news to the dates they were given (:func:`news_until`): pandas keeps the *whole*
attached frame on every slice.

**How the synthetic news is made informative.** For each headline, with
probability ``strength`` its tone matches the sign of the ticker's *next*
``horizon``-day return relative to the cross-section; otherwise the tone is random.
The **generator** looks at future prices so that the news carries information,
the way real news carries information the market has not fully priced yet. A
**strategy** sees only headlines dated up to the day it trades. Set
``strength=0`` for a null panel where the news is pure noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide

POSITIVE = (
    "beats earnings estimates",
    "raises full-year guidance",
    "wins major contract",
    "is upgraded by analysts",
    "reports record revenue",
    "announces share buyback",
    "expands profit margins",
)
NEGATIVE = (
    "misses earnings estimates",
    "cuts full-year guidance",
    "loses key customer",
    "is downgraded by analysts",
    "faces regulatory probe",
    "reports widening losses",
    "recalls flagship product",
)
# Negative news written with positive words: a bag-of-words trap.
NEGATED = (
    "fails to beat earnings estimates",
    "does not raise guidance",
    "no longer expects record revenue",
    "will not expand margins",
)
NEUTRAL = (
    "schedules annual meeting",
    "files quarterly report",
    "names new board member",
    "relocates headquarters",
    "presents at industry conference",
)
TEMPLATES = (
    "{t} {p}",
    "Update: {t} {p}",
    "{t} {p}, sources say",
    "Report: {t} {p}",
)


def synthetic_news(
    prices: pd.DataFrame,
    seed: int = 0,
    p_news: float = 0.10,
    strength: float = 0.20,
    horizon: int = 5,
    p_negated: float = 0.3,
) -> pd.DataFrame:
    """Headlines for a synthetic price panel. See the module docstring for the design.

    ``p_news`` is the chance a (date, ticker) gets a headline; ``p_negated`` the
    share of negative headlines phrased as a negated positive ("fails to beat").
    Includes a ``truth`` column (+1 / 0 / -1, the tone the generator chose) for
    diagnostics only: strategies must not read it.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    rng = np.random.default_rng([seed, 7])
    close = to_wide(prices, "close")
    fwd = close.shift(-horizon) / close - 1.0
    excess = fwd.sub(fwd.mean(axis=1), axis=0)  # NaN in the last `horizon` rows

    has_news = rng.uniform(size=close.shape) < p_news
    informative = rng.uniform(size=close.shape) < strength
    random_tone = rng.choice([-1, 0, 1], size=close.shape)
    planted = np.sign(excess.to_numpy())
    tone = np.where(informative & ~np.isnan(planted), planted, random_tone).astype(int)

    rows = []
    di, ti = np.nonzero(has_news)
    for d, t in zip(di, ti, strict=True):
        ticker = close.columns[t]
        s = tone[d, t]
        if s > 0:
            phrase = POSITIVE[rng.integers(len(POSITIVE))]
        elif s < 0:
            pool = NEGATED if rng.uniform() < p_negated else NEGATIVE
            phrase = pool[rng.integers(len(pool))]
        else:
            phrase = NEUTRAL[rng.integers(len(NEUTRAL))]
        template = TEMPLATES[rng.integers(len(TEMPLATES))]
        rows.append((close.index[d], ticker, template.format(t=ticker, p=phrase), s))
    news = pd.DataFrame(rows, columns=["date", "ticker", "headline", "truth"])
    return news.sort_values(["date", "ticker"]).reset_index(drop=True)


def attach_news(prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``prices`` carrying ``news`` in ``attrs["news"]``."""
    missing = {"date", "ticker", "headline"} - set(news.columns)
    if missing:
        raise ValueError(f"news is missing columns {sorted(missing)}")
    out = prices.copy()
    out.attrs["news"] = news
    return out


def news_until(data: pd.DataFrame) -> pd.DataFrame:
    """The attached news, cut to dates within ``data`` (never later than its last date)."""
    news = data.attrs.get("news")
    if news is None:
        raise ValueError(
            "no news attached to these prices: use attach_news(prices, news), "
            "or DATA DEMO in the terminal (synthetic news comes with it)"
        )
    last = data.index.get_level_values("date").max()
    return news[pd.to_datetime(news["date"]) <= last]


def synthetic_prices_with_news(seed: int = 0, **news_kwargs) -> pd.DataFrame:
    """The synthetic-plus panel (30 stocks, 3 pairs, a GARCH index) with headlines attached."""
    from quantskills.data.synthetic import synthetic_prices

    prices = synthetic_prices(seed=seed, n_pairs=3, garch_index=True)
    return attach_news(prices, synthetic_news(prices, seed=seed, **news_kwargs))
