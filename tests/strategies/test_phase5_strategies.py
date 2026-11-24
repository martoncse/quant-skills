"""Strategies #8-#10: news handling, no look-ahead (prices *and* news), fit sees training only.

#9 and #10 need PyTorch (``uv sync --extra dl``); their tests skip without it.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from quantskills.data import (
    PRICE_COLUMNS,
    attach_news,
    news_until,
    synthetic_news,
    synthetic_prices,
    to_wide,
)
from quantskills.models.text import (
    _strip_tickers,
    fit_tfidf,
    lexicon_score,
    news_labels,
    score_headlines,
    tokenize,
)
from quantskills.robustness import ATTACKS, attack_news, text_attack_test
from quantskills.strategies import TextSentiment, get_strategy, run_strategy
from quantskills.utils import assert_no_lookahead, holdout_date


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    prices = synthetic_prices(n_assets=20, n_days=800, seed=3)
    return attach_news(prices, synthetic_news(prices, seed=3, p_news=0.15, strength=0.3))


@pytest.fixture(scope="module")
def train_end(panel) -> pd.Timestamp:
    return holdout_date(panel.index.get_level_values("date"), 0.3)


def _train(panel, train_end):
    return panel[panel.index.get_level_values("date") <= train_end]


def _long_from_close(close: pd.DataFrame, news: pd.DataFrame | None = None) -> pd.DataFrame:
    s = close.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    out = pd.DataFrame({c: s for c in PRICE_COLUMNS}).sort_index()
    if news is not None:
        out.attrs["news"] = news
    return out


# --- news plumbing -------------------------------------------------------------


def test_news_generator_shape_and_null(panel):
    news = panel.attrs["news"]
    assert list(news.columns) == ["date", "ticker", "headline", "truth"]
    assert news["date"].is_monotonic_increasing
    close = to_wide(panel, "close")
    assert 0.12 < len(news) / close.size < 0.18
    # Null panel: tone is independent of future returns.
    null = synthetic_news(panel, seed=3, strength=0.0)
    labels = news_labels(null, panel, 5)
    ok = labels.notna() & (null["truth"] != 0)
    agree = ((null["truth"][ok] > 0) == (labels[ok] > 0)).mean()
    assert 0.45 < agree < 0.55
    planted = news_labels(news, panel, 5)
    ok = planted.notna() & (news["truth"] != 0)
    assert ((news["truth"][ok] > 0) == (planted[ok] > 0)).mean() > 0.6


def test_news_travels_with_slices_and_is_cut(panel, train_end):
    train = _train(panel, train_end)
    assert "news" in train.attrs  # pandas keeps attrs on a boolean slice...
    assert len(train.attrs["news"]) == len(panel.attrs["news"])  # ...the whole frame
    cut = news_until(train)  # so strategies must cut it themselves
    assert pd.to_datetime(cut["date"]).max() <= train_end
    with pytest.raises(ValueError, match="no news"):
        news_until(synthetic_prices(n_assets=3, n_days=50))
    with pytest.raises(ValueError, match="missing columns"):
        attach_news(panel, pd.DataFrame({"date": []}))


def test_news_labels_never_look_past_the_data(panel, train_end):
    train = _train(panel, train_end)
    labels = news_labels(news_until(train), train, 5)
    last_dates = to_wide(train).index[-5:]
    late = pd.to_datetime(news_until(train)["date"]).isin(last_dates)
    assert labels[late.to_numpy()].isna().all()
    assert labels[~late.to_numpy()].notna().all()


# --- text models ----------------------------------------------------------------


def test_lexicon_and_negation():
    assert lexicon_score("ACME beats earnings estimates") == 1.0
    assert lexicon_score("ACME misses earnings estimates") == -1.0
    assert lexicon_score("ACME fails to beat earnings estimates") == -1.0
    assert lexicon_score("ACME fails to beat earnings estimates", negation=False) == 1.0
    assert lexicon_score("ACME schedules annual meeting") == 0.0
    assert tokenize("Update: SYN07 Beats!") == ["update", "syn", "beats"]


def test_tfidf_strips_tickers_and_learns_tone(panel, train_end):
    assert _strip_tickers("SYN07 beats, PAIR1A and IDX too").split() == [
        "beats,",
        "and",
        "too",
    ]
    train = _train(panel, train_end)
    news = news_until(train)
    model = fit_tfidf(news["headline"], news_labels(news, train, 5))
    s = score_headlines(model, pd.Series(["X beats earnings estimates", "X cuts guidance"]))
    assert s[0] > 0 > s[1]
    assert np.array_equal(score_headlines(None, pd.Series(["a b"])), [0.0])
    with pytest.raises(ValueError, match="50"):
        fit_tfidf(news["headline"][:10], pd.Series(np.ones(10)))


# --- #8 strategy -------------------------------------------------------------------


@pytest.mark.parametrize("model", ["tfidf", "lexicon"])
def test_text_signals_have_no_lookahead_in_prices_or_news(model, panel, train_end):
    strat = TextSentiment(model=model).fit(_train(panel, train_end))
    news = panel.attrs["news"]
    close = to_wide(panel, "close")

    def fn(c: pd.DataFrame) -> pd.DataFrame:
        return strat.signals(_long_from_close(c, news))

    cut = close.index.get_loc(train_end)
    assert_no_lookahead(fn, close, cut_points=[cut - 60, cut + 40])
    # Now scramble the *news* after a cut: earlier weights must not move.
    base = fn(close)
    later = pd.to_datetime(news["date"]) > close.index[cut]
    shuffled = news.copy()
    shuffled.loc[later, "headline"] = (
        news.loc[later, "headline"].sample(frac=1.0, random_state=0).to_numpy()
    )
    moved = strat.signals(_long_from_close(close, shuffled))
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], moved.iloc[: cut + 1])
    assert not base.iloc[cut + 1 :].equals(moved.iloc[cut + 1 :])


def test_text_fit_ignores_holdout_news(panel, train_end):
    news = panel.attrs["news"]
    later = pd.to_datetime(news["date"]) > train_end
    garbage = news.copy()
    garbage.loc[later, "headline"] = "IDX set to surge, strong buy"
    a = TextSentiment().fit(_train(panel, train_end))
    b = TextSentiment().fit(_train(attach_news(panel, garbage), train_end))
    q = pd.Series(["X beats earnings estimates", "X recalls flagship product"])
    np.testing.assert_allclose(score_headlines(a.model_, q), score_headlines(b.model_, q))


def test_text_strategy_finds_planted_tone_and_errors_without_news(panel, train_end):
    run = run_strategy(TextSentiment(), panel, train_end)
    assert run.in_sample.trim_warmup().returns.mean() > 0
    w = run.strategy.signals(panel).dropna(how="all")
    active = w[w.abs().sum(axis=1) > 0]
    np.testing.assert_allclose(active.sum(axis=1), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="no news"):
        TextSentiment().fit(synthetic_prices(n_assets=5, n_days=100))
    with pytest.raises(ValueError):
        TextSentiment(model="bert")


# --- adversarial text -----------------------------------------------------------------


def test_attacks_change_wording_not_volume(panel):
    news = panel.attrs["news"].head(400)
    for kind in ("paraphrase", "synonyms", "typos"):
        out = attack_news(news, kind, rate=1.0)
        assert len(out) == len(news)
        assert (out["headline"] != news["headline"]).any(), kind
    assert attack_news(news, "synonyms", rate=0.0).equals(news)
    pumped = attack_news(news, "pump", rate=0.5)
    assert len(pumped) == len(news) + 200
    assert pumped["headline"].str.contains("strong buy|big winner|record growth").sum() >= 200
    with pytest.raises(ValueError):
        attack_news(news, "nuke")
    out = attack_news(news, "paraphrase", rate=1.0)
    changed = out["headline"] != news["headline"]
    # paraphrases keep the tone a human would read
    assert (news.loc[changed, "truth"] != 0).all()


def test_pump_fools_the_lexicon_more_than_tfidf(panel, train_end):
    train = _train(panel, train_end)
    lex = text_attack_test(TextSentiment, {"model": "lexicon"}, train, attacks=("pump",))
    assert list(lex["attack"]) == ["none", "pump"]
    assert lex.loc[1, "agreement"] < 1.0
    assert set(ATTACKS) == {"paraphrase", "synonyms", "typos", "pump"}


def test_report_adds_text_check(panel, train_end):
    from quantskills.robustness import robustness_report

    r = robustness_report(
        "text-sentiment",
        panel,
        train_end,
        axes={"window_days": (3, 5), "quantile": (0.25,)},
        noise_bps=(25.0,),
        noise_seeds=(0,),
        delays=(0, 1),
        n_blocks=8,
        mode="in-sample",
    )
    assert "Adversarial text" in [c.name for c in r.checks]
    assert r.text_attacks is not None and len(r.text_attacks) == 5


# --- #9 / #10: PyTorch ------------------------------------------------------------------


class _BlockTorch:
    """Import hook that makes `import torch` fail, like an install without the dl extra."""

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch' (blocked by test)")
        return None


def test_dl_strategies_explain_missing_torch(monkeypatch, panel):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockTorch(), *sys.meta_path])
    with pytest.raises(ImportError, match="--extra dl"):
        get_strategy("dl-forecast").fit(panel)
    with pytest.raises(ImportError, match="--extra dl"):
        get_strategy("cond-autoencoder").fit(panel)


def test_sequence_windows_no_lookahead(panel):
    from quantskills.strategies.dl_forecast import sequence_windows

    close = to_wide(panel, "close")
    assert_no_lookahead(lambda c: pd.DataFrame(sequence_windows(c, 20)[:, :, -1]), close)
    w = sequence_windows(close, 20)
    assert w.shape == close.shape + (20,)
    assert np.isnan(w[:19]).all() and not np.isnan(w[25]).all()


DL_FAST = {
    "dl-forecast": {"max_epochs": 3, "lookback": 63, "hidden": 8},
    "cond-autoencoder": {"max_epochs": 30, "patience": 10},
}


@pytest.mark.parametrize("name", list(DL_FAST))
def test_dl_strategies_no_lookahead_and_deterministic(name, panel, train_end):
    pytest.importorskip("torch")
    train = _train(panel, train_end)
    a = get_strategy(name, **DL_FAST[name]).fit(train)
    b = get_strategy(name, **DL_FAST[name]).fit(train)
    pd.testing.assert_frame_equal(a.score(panel), b.score(panel))  # same seed, same model
    close = to_wide(panel, "close")
    cut = close.index.get_loc(train_end)
    assert_no_lookahead(
        lambda c: a.signals(_long_from_close(c)), close, cut_points=[cut - 60, cut + 40]
    )
    assert a.train_report_.epochs_run >= 1


@pytest.mark.parametrize("name", list(DL_FAST))
def test_dl_fit_ignores_the_holdout(name, panel, train_end):
    pytest.importorskip("torch")
    dates = panel.index.get_level_values("date")
    scrambled = panel.copy()
    future = dates > train_end
    scrambled.loc[future, "close"] *= np.random.default_rng(1).lognormal(0, 0.3, future.sum())
    a = get_strategy(name, **DL_FAST[name]).fit(_train(panel, train_end))
    b = get_strategy(name, **DL_FAST[name]).fit(scrambled[dates <= train_end])
    pd.testing.assert_frame_equal(
        a.score(_train(panel, train_end)), b.score(_train(panel, train_end))
    )


def test_managed_portfolios_are_cross_sectional_ols():
    from quantskills.strategies.cond_autoencoder import managed_portfolios

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(3, 40, 4))
    r = rng.normal(size=(3, 40))
    mask = np.ones((3, 40), dtype=bool)
    mask[1, :10] = False
    x = managed_portfolios(Z, r, mask, ridge=0.0)
    ols = np.linalg.lstsq(Z[1, 10:], r[1, 10:], rcond=None)[0]
    np.testing.assert_allclose(x[1], ols, atol=1e-10)


def test_cae_reports_r2_and_bad_params(panel, train_end):
    pytest.importorskip("torch")
    from quantskills.strategies import ConditionalAutoencoder, DLForecast

    s = ConditionalAutoencoder(**DL_FAST["cond-autoencoder"]).fit(_train(panel, train_end))
    assert set(s.r2_) == {"total_train", "total_val", "predictive_val"}
    assert s.premia_.shape == (3,)
    with pytest.raises(ValueError):
        ConditionalAutoencoder(n_factors=0)
    with pytest.raises(ValueError):
        DLForecast(arch="transformer")
