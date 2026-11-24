"""Strategy #9: deep-learning return forecast (MLP / 1-D CNN / LSTM). Needs ``--extra dl``.

Input per (date, ticker): the last ``lookback`` (252) daily returns in excess of the
cross-sectional mean, divided by the ticker's trailing volatility, so every
sequence has a comparable scale. Target: the cross-sectional rank of the next
``horizon``-day return (as in #2 and #6). A quantile long-short book trades the
ranking.

Training (inside ``fit``, on the training window only):
* samples every ``sample_every`` dates, to cut the overlap between 21-day labels;
* the last ``val_frac`` of training dates is the validation set, separated from the
  fitting dates by ``horizon`` days so no label straddles the boundary;
* early stopping on validation mean daily IC; the best epoch's weights are kept.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.features.panel import label_panel
from quantskills.models.nn import (
    ARCHS,
    make_sequence_net,
    mean_daily_ic,
    require_torch,
    seed_everything,
    train_early_stopping,
)
from quantskills.strategies._weights import quantile_long_short
from quantskills.strategies.base import Strategy


def sequence_windows(close: pd.DataFrame, lookback: int) -> np.ndarray:
    """``(T, N, lookback)`` array; row t holds scaled excess returns for days t-lookback+1..t.

    NaN wherever the window is incomplete. Uses data up to and including t only.
    """
    r = close.pct_change(fill_method=None)
    ex = r.sub(r.mean(axis=1), axis=0)
    vol = ex.rolling(lookback, min_periods=lookback // 2).std()
    a = ex.to_numpy(dtype=float)
    t_len, n = a.shape
    out = np.full((t_len, n, lookback), np.nan)
    if t_len >= lookback:
        win = np.lib.stride_tricks.sliding_window_view(a, lookback, axis=0)  # (T-L+1, N, L)
        out[lookback - 1 :] = win
    scale = vol.to_numpy(dtype=float)[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.clip(out / scale, -5.0, 5.0)
    return out


class DLForecast(Strategy):
    """#9: MLP / 1-D CNN / LSTM on a year of scaled returns, early-stopped. Needs --extra dl."""

    name = "dl-forecast"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "arch": "cnn",  # "mlp", "cnn" or "lstm"
            "lookback": 252,  # long enough to see 12-month momentum
            "horizon": 21,
            "hidden": 16,
            "dropout": 0.1,
            "lr": 1e-3,
            "max_epochs": 30,
            "patience": 5,
            "batch_size": 256,
            "sample_every": 5,
            "val_frac": 0.2,
            "rebalance_days": 21,
            "quantile": 0.2,
            "gross": 1.0,
            "min_assets": 10,
            "seed": 0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["arch"] not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        if not 0 < self.params["val_frac"] < 0.5:
            raise ValueError("val_frac must be in (0, 0.5)")
        self.net_: Any = None
        self.train_report_: Any = None

    def _xy(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        close = to_wide(data, "close")
        X = sequence_windows(close, p["lookback"])
        y = (
            label_panel(close, p["horizon"], "rank")
            .unstack()
            .reindex(index=close.index, columns=close.columns)
        )
        rows = np.arange(0, len(close), p["sample_every"])
        Xs, ys, gs = X[rows], y.to_numpy(dtype=float)[rows], np.repeat(rows, close.shape[1])
        Xs, ys = Xs.reshape(-1, p["lookback"]), ys.reshape(-1)
        ok = ~np.isnan(Xs).any(axis=1) & ~np.isnan(ys)
        return Xs[ok], ys[ok], gs[ok]

    def fit(self, data: pd.DataFrame) -> Self:
        p = self.params
        torch = seed_everything(p["seed"])
        X, y, g = self._xy(data)
        if len(X) < 500:
            raise ValueError(f"only {len(X)} training sequences; need more history")
        dates = np.unique(g)
        cut = dates[int(len(dates) * (1 - p["val_frac"]))]
        fit_m = g < cut - p["horizon"]  # purge: no fitting label overlaps validation
        val_m = g >= cut
        Xt, yt = torch.tensor(X[fit_m], dtype=torch.float32), torch.tensor(y[fit_m]).float()
        Xv = torch.tensor(X[val_m], dtype=torch.float32)
        yv, gv = y[val_m], g[val_m]

        net = make_sequence_net(p["arch"], p["lookback"], p["hidden"], p["dropout"])
        gen = torch.Generator().manual_seed(p["seed"])
        loss_fn = torch.nn.MSELoss()

        def step(opt: Any, _train: bool) -> float:
            perm = torch.randperm(len(Xt), generator=gen)
            total = 0.0
            for i in range(0, len(Xt), p["batch_size"]):
                idx = perm[i : i + p["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(net(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            return total / len(Xt)

        def validate(model: Any) -> float:
            return mean_daily_ic(model(Xv).squeeze(1).numpy(), yv, gv)

        self.train_report_ = train_early_stopping(
            net, step, validate, p["max_epochs"], p["patience"], p["lr"]
        )
        self.net_ = net.eval()
        self.is_fitted = True
        return self

    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        torch = require_torch()
        close = to_wide(data, "close")
        X = sequence_windows(close, self.params["lookback"])
        flat = X.reshape(-1, X.shape[2])
        ok = ~np.isnan(flat).any(axis=1)
        pred = np.full(len(flat), np.nan)
        if ok.any():
            with torch.no_grad():
                pred[ok] = self.net_(torch.tensor(flat[ok], dtype=torch.float32)).squeeze(1).numpy()
        return pd.DataFrame(pred.reshape(X.shape[:2]), index=close.index, columns=close.columns)

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return quantile_long_short(
            self.score(data), p["quantile"], p["gross"], p["rebalance_days"], p["min_assets"]
        )
