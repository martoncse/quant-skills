"""Synthetic market histories for stress tests (strategy #11).

A backtest is one path through history. To ask "how would this strategy do on
*other* histories with the same risk?", fit a generator to the returns and draw
many alternative paths:

* ``gaussian``: i.i.d. multivariate normal with the sample mean and covariance.
  Keeps correlations; loses fat tails, volatility clustering and every signal.
* ``bootstrap``: stationary block bootstrap (Politis & Romano, 1994) of whole days.
  Keeps same-day correlations, fat tails and short-range dependence (blocks average
  ``mean_block`` days); breaks anything slower, such as persistent drift or
  cointegrated *levels*. A useful null for slow signals.
* ``timegan``: windows from a TimeGAN (``quantskills.models.timegan``, dl extra) trained
  on each window's price path (see :func:`path_windows`), chained end to end. Nothing
  links one window to the next.

``fidelity_report`` scores how realistic each generator is (stylized facts, a
classifier trying to tell real from fake, train-on-synthetic-test-on-real), and
``stress_test`` reruns a strategy on many synthetic histories to give a Sharpe
*distribution* and where the real result sits in it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quantskills.backtest.engine import CostModel
from quantskills.data.loader import PRICE_COLUMNS, to_wide
from quantskills.eval.metrics import sharpe_ratio
from quantskills.strategies import Strategy, run_strategy
from quantskills.utils.splits import holdout_date

GENERATORS = ("gaussian", "bootstrap", "timegan")


# --- generators ------------------------------------------------------------------


def gaussian_paths(returns: pd.DataFrame, n_paths: int, seed: int = 0) -> list[pd.DataFrame]:
    r = returns.dropna()
    rng = np.random.default_rng([seed, 21])
    mu, cov = r.mean().to_numpy(), np.cov(r.to_numpy(), rowvar=False)
    return [
        pd.DataFrame(rng.multivariate_normal(mu, cov, len(r)), index=r.index, columns=r.columns)
        for _ in range(n_paths)
    ]


def stationary_bootstrap_index(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Row indices: blocks of geometric length (mean ``mean_block``), random starts, wrapping."""
    idx = np.empty(n, dtype=int)
    i = int(rng.integers(n))
    for t in range(n):
        idx[t] = i
        i = int(rng.integers(n)) if rng.uniform() < 1.0 / mean_block else (i + 1) % n
    return idx


def bootstrap_paths(
    returns: pd.DataFrame, n_paths: int, seed: int = 0, mean_block: float = 20.0
) -> list[pd.DataFrame]:
    r = returns.dropna()
    rng = np.random.default_rng([seed, 22])
    a = r.to_numpy()
    return [
        pd.DataFrame(
            a[stationary_bootstrap_index(len(a), mean_block, rng)], index=r.index, columns=r.columns
        )
        for _ in range(n_paths)
    ]


def path_windows(returns: pd.DataFrame, seq_len: int, stride: int = 1) -> np.ndarray:
    """Windows of cumulative log returns (a price path starting at 0), as TimeGAN expects.

    TimeGAN's supervisor learns to predict the next latent step. On near-white-noise
    returns its best guess is the mean, which squashes generated returns to almost
    nothing; price paths are smooth and predictable, like the stock *prices* in the
    TimeGAN paper.
    """
    from quantskills.models.timegan import make_windows

    w = make_windows(np.log1p(returns.dropna().to_numpy()), seq_len, stride)
    return w.cumsum(axis=1)


def _paths_to_returns(paths: np.ndarray) -> np.ndarray:
    first = np.zeros_like(paths[:, :1])
    return np.expm1(np.diff(np.concatenate([first, paths], axis=1), axis=1))


def timegan_paths(
    returns: pd.DataFrame, n_paths: int, seed: int = 0, model: Any = None, **cfg: Any
) -> list[pd.DataFrame]:
    """Fit a TimeGAN on window price paths (or reuse ``model``) and chain sampled windows."""
    from quantskills.models.timegan import TimeGAN, TimeGANConfig

    r = returns.dropna()
    if model is None:
        c = TimeGANConfig(seed=seed, **cfg)
        model = TimeGAN(c).fit(path_windows(r, c.seq_len))
    L = model.cfg.seq_len
    per_path = -(-len(r) // L)
    out = []
    for k in range(n_paths):
        w = _paths_to_returns(model.sample(per_path, seed=seed * 1000 + k))
        out.append(
            pd.DataFrame(w.reshape(-1, r.shape[1])[: len(r)], index=r.index, columns=r.columns)
        )
    return out


def synthetic_paths(
    returns: pd.DataFrame, method: str, n_paths: int, seed: int = 0, **kwargs: Any
) -> list[pd.DataFrame]:
    """``n_paths`` synthetic return histories shaped like ``returns`` (date x ticker)."""
    if method == "gaussian":
        return gaussian_paths(returns, n_paths, seed)
    if method == "bootstrap":
        return bootstrap_paths(returns, n_paths, seed, **kwargs)
    if method == "timegan":
        return timegan_paths(returns, n_paths, seed, **kwargs)
    raise ValueError(f"method must be one of {GENERATORS}")


def returns_to_panel(returns: pd.DataFrame, start_price: float = 100.0) -> pd.DataFrame:
    """Simple returns (date x ticker) -> long OHLCV panel (open = high = low = close)."""
    close = start_price * (1.0 + returns.clip(lower=-0.95)).cumprod()
    s = close.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    panel = pd.DataFrame({c: s for c in PRICE_COLUMNS})
    panel["volume"] = 1e6
    return panel.sort_index()


# --- fidelity --------------------------------------------------------------------


def _acf(x: np.ndarray, lag: int) -> float:
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def stylized_facts(returns: pd.DataFrame) -> dict[str, float]:
    """Averages across tickers of the textbook properties of daily returns."""
    r = returns.dropna().to_numpy()
    cols = range(r.shape[1])
    corr = np.corrcoef(r, rowvar=False)
    off = corr[~np.eye(len(corr), dtype=bool)] if r.shape[1] > 1 else np.array([np.nan])
    return {
        "ann_vol": float(np.mean(r.std(axis=0, ddof=1)) * np.sqrt(252)),
        "skew": float(np.mean([stats.skew(r[:, j]) for j in cols])),
        "excess_kurtosis": float(np.mean([stats.kurtosis(r[:, j]) for j in cols])),
        "acf1_returns": float(np.nanmean([_acf(r[:, j], 1) for j in cols])),
        "acf_abs_1to5": float(
            np.nanmean([_acf(np.abs(r[:, j]), k) for j in cols for k in range(1, 6)])
        ),  # volatility clustering
        "mean_pairwise_corr": float(np.mean(off)),
    }


def _window_features(w: np.ndarray) -> np.ndarray:
    """Per window: per-asset log volatility, kurtosis, |r| lag-1 autocorrelation, mean corr."""
    feats = []
    for x in w:
        sd = np.log(x.std(axis=0) + 1e-12)
        kurt = stats.kurtosis(x, axis=0)
        ab = np.abs(x)
        ac = np.array([_acf(ab[:, j], 1) for j in range(x.shape[1])])
        c = np.corrcoef(x, rowvar=False)
        mc = np.nanmean(c[~np.eye(len(c), dtype=bool)]) if x.shape[1] > 1 else 0.0
        feats.append(np.r_[sd, kurt, np.nan_to_num(ac), mc])
    return np.nan_to_num(np.asarray(feats))


def discriminative_score(real: np.ndarray, fake: np.ndarray, seed: int = 0) -> float:
    """|accuracy - 0.5| of a classifier telling real windows from fake (0 = indistinguishable).

    L2 logistic regression on standardised window statistics, 5-fold cross-validated
    (samples are small: a few dozen windows per class). Pass **non-overlapping** real
    windows, or the classifier sees near-copies in training and test. With ~50 windows
    per class, differences below ~0.1 are noise.
    """
    n = min(len(real), len(fake))
    rng = np.random.default_rng(seed)
    fake = fake[rng.permutation(len(fake))[:n]]
    X = np.vstack([_window_features(real[:n]), _window_features(fake)])
    y = np.r_[np.ones(n), np.zeros(n)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=folds, scoring="accuracy").mean()
    return float(abs(acc - 0.5))


def _vol_xy(w: np.ndarray, lags: int = 5) -> tuple[np.ndarray, np.ndarray]:
    a = np.abs(w)  # (n, L, d)
    X = np.stack([a[:, i : i + lags, :] for i in range(a.shape[1] - lags)], axis=1)
    y = a[:, lags:, :]
    return X.reshape(-1, lags), y.reshape(-1)


def predictive_score(real_train: np.ndarray, real_test: np.ndarray, fake: np.ndarray) -> dict:
    """Train-on-synthetic, test-on-real: predict |r| from the previous 5 |r| (a vol forecast).

    Compare its error with training on real data (``trtr``). Close to ``trtr`` means the
    synthetic data carry the same volatility dynamics.
    """
    Xt, yt = _vol_xy(real_test)
    out = {}
    for name, w in (("tstr", fake), ("trtr", real_train)):
        X, y = _vol_xy(w)
        out[f"{name}_mae"] = float(np.mean(np.abs(Ridge(1.0).fit(X, y).predict(Xt) - yt)))
    return out


def fidelity_report(
    returns: pd.DataFrame,
    methods: Sequence[str] = GENERATORS,
    n_paths: int = 5,
    seq_len: int = 24,
    seed: int = 0,
    timegan_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """One row per generator (plus ``real``): stylized facts, discriminative and TSTR scores."""
    from quantskills.models.timegan import make_windows

    r = returns.dropna()
    half = len(r) // 2
    real_tr = make_windows(r.to_numpy()[:half], seq_len, stride=seq_len)
    real_te = make_windows(r.to_numpy()[half:], seq_len, stride=seq_len)
    real_all = make_windows(r.to_numpy(), seq_len, stride=seq_len)
    rows = {"real": {**stylized_facts(r), "discriminative": np.nan}}
    rows["real"].update(predictive_score(real_tr, real_te, real_tr))
    for m in methods:
        kw = dict(timegan_kwargs or {}, seq_len=seq_len) if m == "timegan" else {}
        paths = synthetic_paths(r, m, n_paths, seed, **kw)
        facts = pd.DataFrame([stylized_facts(p) for p in paths]).mean().to_dict()
        fake = np.concatenate([make_windows(p.to_numpy(), seq_len, stride=seq_len) for p in paths])
        rows[m] = {
            **facts,
            "discriminative": discriminative_score(real_all, fake, seed),
            **predictive_score(real_tr, real_te, fake),
        }
    return pd.DataFrame(rows).T


# --- stress test -------------------------------------------------------------------


def stress_test(
    strategy: Strategy,
    prices: pd.DataFrame,
    method: str = "bootstrap",
    n_paths: int = 20,
    seed: int = 0,
    test_fraction: float = 0.3,
    costs: CostModel | None = None,
    **gen_kwargs: Any,
) -> pd.DataFrame:
    """Run ``strategy`` on the real prices and on ``n_paths`` synthetic histories.

    Each history is split like the real one (last ``test_fraction`` held out). Pass
    the **training window** as ``prices`` to keep the real hold-out untouched.
    Returns one row per path (``path = -1`` is the real data) with in-sample and
    hold-out Sharpe; see :func:`stress_summary`.
    """
    if getattr(strategy, "uses_news", False):
        raise ValueError("synthetic price paths carry no news; stress-test #8 another way")
    cls, params = type(strategy), dict(strategy.params)
    close = to_wide(prices, "close")
    returns = close.pct_change(fill_method=None).iloc[1:]

    def one(panel: pd.DataFrame) -> tuple[float, float]:
        te = holdout_date(panel.index.get_level_values("date"), test_fraction)
        try:
            run = run_strategy(cls(**params), panel, te, costs)
        except ValueError:  # e.g. no cointegrated pair in a synthetic path
            return 0.0, 0.0
        is_sr = sharpe_ratio(run.in_sample.trim_warmup().returns)
        oos_sr = sharpe_ratio(run.out_of_sample.trim_warmup().returns)
        return float(np.nan_to_num(is_sr)), float(np.nan_to_num(oos_sr))

    rows = [(-1, *one(prices))]
    paths = synthetic_paths(returns, method, n_paths, seed, **gen_kwargs)
    for k, p in enumerate(paths):
        rows.append((k, *one(returns_to_panel(p, 100.0))))
    return pd.DataFrame(rows, columns=["path", "is_sharpe", "oos_sharpe"])


def stress_summary(table: pd.DataFrame) -> dict[str, float]:
    """Synthetic Sharpe distribution and the real result's percentile in it."""
    real = table[table["path"] == -1].iloc[0]
    syn = table[table["path"] >= 0]
    out: dict[str, float] = {"n_paths": float(len(syn))}
    for col in ("is_sharpe", "oos_sharpe"):
        v = syn[col].to_numpy()
        out[f"{col}_real"] = float(real[col])
        out[f"{col}_median"] = float(np.median(v))
        out[f"{col}_p05"] = float(np.quantile(v, 0.05))
        out[f"{col}_p95"] = float(np.quantile(v, 0.95))
        out[f"{col}_pct_real"] = float((v < real[col]).mean())
    return out
