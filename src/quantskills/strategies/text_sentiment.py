"""Strategy #8: news sentiment long-short.

Score every headline (lexicon, or TF-IDF + logistic fitted on the training
window), sum each ticker's scores over the last ``window_days`` days, and trade a
quantile long-short book on that sum among tickers with recent news.

News comes attached to the prices (``quantskills.data.news.attach_news``); the
strategy only ever reads headlines dated on or before the row it is computing.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.data.news import news_until
from quantskills.models.text import fit_tfidf, news_labels, score_headlines
from quantskills.strategies._weights import quantile_long_short
from quantskills.strategies.base import Strategy

MODELS = ("tfidf", "lexicon")


class TextSentiment(Strategy):
    """#8: long-short on recent headline tone (TF-IDF + logistic, or a lexicon)."""

    name = "text-sentiment"
    uses_news: ClassVar[bool] = True

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "model": "tfidf",  # or "lexicon" (no fitting)
            "horizon": 5,  # label horizon for the TF-IDF model, days
            "window_days": 5,  # sum headline tone over this many past days
            "quantile": 0.25,
            "rebalance_days": 5,
            "gross": 1.0,
            "min_assets": 6,
            "C": 1.0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["model"] not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.params["window_days"] < 1:
            raise ValueError("window_days must be >= 1")
        self.model_: Any = None
        self.n_train_headlines_ = 0

    def fit(self, data: pd.DataFrame) -> Self:
        news = news_until(data)
        self.n_train_headlines_ = len(news)
        if self.params["model"] == "tfidf":
            labels = news_labels(news, data, self.params["horizon"])
            self.model_ = fit_tfidf(news["headline"], labels, C=self.params["C"])
        self.is_fitted = True
        return self

    def tone(self, data: pd.DataFrame) -> pd.DataFrame:
        """Summed headline tone per (date, ticker) over the trailing window; NaN = no news."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        close = to_wide(data, "close")
        news = news_until(data)
        news = news[news["ticker"].isin(close.columns)]
        scored = pd.DataFrame(
            {
                "date": pd.to_datetime(news["date"]),
                "ticker": news["ticker"],
                "s": score_headlines(self.model_, news["headline"]),
            }
        )
        daily = scored.pivot_table(index="date", columns="ticker", values="s", aggfunc="sum")
        daily = daily.reindex(index=close.index, columns=close.columns)
        w = self.params["window_days"]
        summed = daily.fillna(0.0).rolling(w, min_periods=1).sum()
        recent = daily.notna().astype(float).rolling(w, min_periods=1).sum() > 0
        return summed.where(recent)

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return quantile_long_short(
            self.tone(data), p["quantile"], p["gross"], p["rebalance_days"], p["min_assets"]
        )


def headline_tone_ic(strategy: TextSentiment, data: pd.DataFrame) -> float:
    """Diagnostic: mean daily rank IC of tone vs the next ``horizon``-day return."""
    tone = strategy.tone(data)
    close = to_wide(data, "close")
    fwd = close.shift(-strategy.params["horizon"]) / close - 1.0
    ics = []
    for d in tone.index:
        a, b = tone.loc[d], fwd.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 5 and a[ok].nunique() > 1:
            ics.append(a[ok].rank().corr(b[ok].rank()))
    return float(np.mean(ics)) if ics else float("nan")
