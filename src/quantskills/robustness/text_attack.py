"""Adversarial stress for news-driven signals.

A sentiment model is fitted on clean headlines, then the live feed is attacked
and we see how much of the signal survives. Each attack keeps the *meaning* a
human reader would take from the news (or, for ``pump``, adds fake news), so a
robust signal should barely move:

* ``paraphrase``: "misses earnings estimates" → "fails to beat earnings estimates".
  Same meaning, written with positive words plus a negator.
* ``synonyms``: tone words swapped for synonyms the model never saw in training
  ("beats" → "tops", "cuts" → "trims").
* ``typos``: one adjacent-letter swap in each tone word ("beats" → "beast"). Humans
  read through it; a vocabulary lookup does not.
* ``pump``: an adversary floods the feed with fake upbeat headlines on random
  tickers (``rate`` x as many as real headlines). Nothing about them is informative.

Run on the **training window**: fit on its first ``fit_frac`` of dates, score the
rest with clean and attacked news. The hold-out is not used.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantskills.backtest.engine import CostModel, run_backtest
from quantskills.data.loader import to_wide
from quantskills.data.news import attach_news, news_until
from quantskills.eval.metrics import sharpe_ratio
from quantskills.strategies.base import Strategy

ATTACKS = ("paraphrase", "synonyms", "typos", "pump")

PARAPHRASES = {
    "misses earnings estimates": "fails to beat earnings estimates",
    "cuts full-year guidance": "does not raise full-year guidance",
    "reports widening losses": "no longer reports record profits",
    "is downgraded by analysts": "is not upgraded by analysts",
    "beats earnings estimates": "does not miss earnings estimates",
    "raises full-year guidance": "no longer cuts full-year guidance",
}
SYNONYMS = {
    "beats": "tops", "raises": "lifts", "wins": "secures", "upgraded": "promoted",
    "record": "all-time", "buyback": "repurchase", "expands": "widens",
    "misses": "trails", "cuts": "trims", "loses": "sheds", "downgraded": "demoted",
    "probe": "inquiry", "losses": "deficits", "recalls": "withdraws", "beat": "top",
}  # fmt: skip
PUMP = (
    "{t} set to surge, strong buy",
    "{t} is the next big winner, insiders say",
    "{t} poised for record growth",
)
_TONE = set(SYNONYMS) | {"raise", "upgrade", "expand"}


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 2))
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def attack_news(news: pd.DataFrame, kind: str, rate: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Return an attacked copy of ``news``. ``rate`` = share of headlines hit (or pump volume)."""
    if kind not in ATTACKS:
        raise ValueError(f"kind must be one of {ATTACKS}")
    rng = np.random.default_rng([seed, 11])
    out = news.copy()
    if kind == "pump":
        n = int(round(rate * len(news)))
        if n == 0 or news.empty:
            return out
        idx = rng.integers(0, len(news), n)  # fake headlines on real (date) slots
        tickers = news["ticker"].unique()
        fake = pd.DataFrame(
            {
                "date": news["date"].to_numpy()[idx],
                "ticker": rng.choice(tickers, n),
            }
        )
        fake["headline"] = [PUMP[rng.integers(len(PUMP))].format(t=t) for t in fake["ticker"]]
        if "truth" in news.columns:
            fake["truth"] = 0
        out = pd.concat([out, fake], ignore_index=True)
        return out.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

    hit = rng.uniform(size=len(out)) < rate
    heads = out["headline"].to_numpy(dtype=object)
    for i in np.flatnonzero(hit):
        h = heads[i]
        if kind == "paraphrase":
            for a, b in PARAPHRASES.items():
                if a in h:
                    h = h.replace(a, b)
                    break
        elif kind == "synonyms":
            h = re.sub(r"[a-z]+", lambda m: SYNONYMS.get(m.group(0), m.group(0)), h)
        else:  # typos
            h = re.sub(
                r"[a-z]+",
                lambda m: _typo(m.group(0), rng) if m.group(0) in _TONE else m.group(0),
                h,
            )
        heads[i] = h
    out["headline"] = heads
    return out


def text_attack_test(
    strategy_cls: type[Strategy],
    params: Mapping[str, Any],
    data: pd.DataFrame,
    attacks: Sequence[str] = ATTACKS,
    rate: float = 0.5,
    seed: int = 0,
    fit_frac: float = 0.6,
    costs: CostModel | None = None,
) -> pd.DataFrame:
    """Sharpe on the last ``1 - fit_frac`` of ``data`` with clean vs attacked news.

    Columns: ``attack``, ``sharpe``, ``kept`` (attacked / clean Sharpe, NaN when the
    clean Sharpe is <= 0) and ``agreement`` (rank correlation of clean vs attacked
    tone, pooled over the scored days).
    """
    dates = data.index.get_level_values("date")
    udates = dates.unique().sort_values()
    split = udates[int(len(udates) * fit_frac)]
    strategy = strategy_cls(**params).fit(data[dates <= split])
    close = to_wide(data, "close")
    clean_news = news_until(data)

    def run(news: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        d = attach_news(data, news)
        tone = strategy.tone(d)  # type: ignore[attr-defined]
        res = run_backtest(strategy.signals(d), close, costs)
        r = res.returns[res.returns.index > split]
        return sharpe_ratio(r), tone[tone.index > split]

    base_sr, base_tone = run(clean_news)
    rows = [{"attack": "none", "sharpe": base_sr, "kept": 1.0, "agreement": 1.0}]
    for kind in attacks:
        sr, tone = run(attack_news(clean_news, kind, rate, seed))
        a, b = base_tone.stack(), tone.stack()
        common = a.index.intersection(b.index)
        agree = float(a[common].rank().corr(b[common].rank())) if len(common) > 2 else np.nan
        kept = sr / base_sr if base_sr > 0 else np.nan
        rows.append({"attack": kind, "sharpe": sr, "kept": kept, "agreement": agree})
    return pd.DataFrame(rows)
