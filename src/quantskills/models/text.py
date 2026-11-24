"""Headline sentiment: a hand-written lexicon baseline and a learned TF-IDF model.

* **Lexicon:** +1 per positive word, -1 per negative word; a negator ("not",
  "fails", "no longer"...) flips the next three words. No fitting; a clear baseline.
* **TF-IDF + logistic regression:** word and word-pair (bigram) counts, weighted
  down when common, fed to a logistic model of "this stock beats the cross-section
  over the next ``horizon`` days". Bigrams let it learn "fails to" from data.

Labels are built from prices **inside the window the model is fitted on**; the last
``horizon`` dates have no label and are dropped, so nothing leaks past the window.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from quantskills.data.loader import to_wide

POSITIVE_WORDS = frozenset(
    "beat beats raise raises raised win wins upgraded upgrade record buyback expand "
    "expands growth strong surge".split()
)
NEGATIVE_WORDS = frozenset(
    "miss misses cut cuts lose loses downgraded downgrade probe losses widening recall "
    "recalls weak plunge".split()
)
NEGATORS = frozenset("not no fails fail never longer".split())
_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def lexicon_score(text: str, negation: bool = True, window: int = 3) -> float:
    """Net tone in [-1, 1]: (pos - neg) / (pos + neg), 0 when no tone words."""
    pos = neg = 0
    flip_left = 0
    for tok in tokenize(text):
        if negation and tok in NEGATORS:
            flip_left = window
            continue
        sign = 1 if tok in POSITIVE_WORDS else -1 if tok in NEGATIVE_WORDS else 0
        if flip_left > 0:
            sign, flip_left = -sign, flip_left - 1
        pos += sign > 0
        neg += sign < 0
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def news_labels(news: pd.DataFrame, prices: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """1 if the ticker beats the cross-sectional mean over the next ``horizon`` days.

    Computed from ``prices`` only: headlines whose window runs past the last
    price date get NaN.
    """
    close = to_wide(prices, "close")
    fwd = close.shift(-horizon) / close - 1.0
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    stacked = excess.stack(future_stack=True)
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(news["date"]), news["ticker"]])
    vals = stacked.reindex(keys).to_numpy()
    out = np.where(np.isnan(vals), np.nan, (vals > 0).astype(float))
    return pd.Series(out, index=news.index, name="label")


def fit_tfidf(headlines: pd.Series, labels: pd.Series, C: float = 1.0, min_df: int = 3) -> Pipeline:
    """TF-IDF (unigrams + bigrams) + L2 logistic regression. Ticker symbols are stripped."""
    ok = labels.notna()
    if ok.sum() < 50 or labels[ok].nunique() < 2:
        raise ValueError("need at least 50 labelled headlines of both classes")
    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2), min_df=min_df, preprocessor=_strip_tickers, token_pattern=r"[a-z]+"
        ),
        LogisticRegression(C=C, max_iter=1000),
    )
    return model.fit(headlines[ok], labels[ok].astype(int))


def _strip_tickers(text: str) -> str:
    # Tickers are upper-case symbols such as SYN07 or AAPL: drop them so the model
    # learns language, not which stock tends to go up.
    return re.sub(r"\b[A-Z][A-Z0-9.^-]{1,}\b", " ", text).lower()


def score_headlines(model: Pipeline | None, headlines: pd.Series) -> np.ndarray:
    """Tone in [-1, 1]: 2 * P(up) - 1 for a fitted model, the lexicon score for ``None``."""
    if len(headlines) == 0:
        return np.zeros(0)
    if model is None:
        return np.array([lexicon_score(h) for h in headlines], dtype=float)
    return 2.0 * model.predict_proba(headlines)[:, 1] - 1.0
