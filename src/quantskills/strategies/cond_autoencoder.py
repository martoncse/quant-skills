"""Strategy #10: conditional autoencoder factor model. Needs ``--extra dl``.

Idea (Gu, Kelly & Xiu, 2021): returns follow a few latent factors, and each stock's
exposure (beta) to them depends on its characteristics.

    r_{i,t+h} = beta(z_{i,t}) . f_{t+h} + e

* ``beta(z)``: a small network from the stock's characteristics at *t* (the panel
  features: momentum, reversal, volatility, trend) to ``n_factors`` exposures.
* ``f``: factors built from *managed portfolios* x = (Z'Z)^-1 Z' r, i.e. the
  cross-sectional regression of returns on characteristics, followed by a linear
  layer. This is the "autoencoder": returns in, a few factors in the middle,
  returns reconstructed out.

Trained to reconstruct ``h``-day returns on the training window (non-overlapping
blocks). To *trade*, we need an expected return known at *t*: beta(z_t) . lambda,
where lambda is the average factor value over the training window (the factor
premium). A quantile long-short book trades that forecast.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.features.panel import ML_FEATURES, feature_panel
from quantskills.models.nn import require_torch, seed_everything, train_early_stopping
from quantskills.strategies._weights import quantile_long_short
from quantskills.strategies.base import Strategy


def _char_tensor(close: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    """``(T, N, P+1)`` characteristics (z-scored panel features + a constant)."""
    fp = feature_panel(close, features)
    cube = np.stack(
        [fp[f].unstack().reindex(index=close.index, columns=close.columns).to_numpy() for f in fp],
        axis=2,
    )
    const = np.ones(cube.shape[:2] + (1,))
    return np.concatenate([cube, const], axis=2)


def managed_portfolios(Z: np.ndarray, r: np.ndarray, mask: np.ndarray, ridge: float = 1e-3):
    """Per date: x = (Z'Z + ridge I)^-1 Z' r over the stocks in ``mask``. Shape (D, P)."""
    d, _, p = Z.shape
    out = np.zeros((d, p))
    for t in range(d):
        m = mask[t]
        if m.sum() <= p:
            continue
        z, y = Z[t, m], r[t, m]
        out[t] = np.linalg.solve(z.T @ z + ridge * np.eye(p), z.T @ y)
    return out


class ConditionalAutoencoder(Strategy):
    """#10: conditional autoencoder; trades beta(characteristics) x factor premia (dl extra)."""

    name = "cond-autoencoder"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "features": tuple(ML_FEATURES),
            "n_factors": 3,
            "hidden": 16,
            "ret_horizon": 5,  # reconstruct non-overlapping 5-day returns
            "l1": 1e-4,
            "lr": 1e-2,  # full-batch: one optimiser step per epoch
            "max_epochs": 600,
            "patience": 60,
            "val_frac": 0.2,
            "rebalance_days": 5,
            "quantile": 0.2,
            "gross": 1.0,
            "min_assets": 10,
            "seed": 0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["n_factors"] < 1:
            raise ValueError("n_factors must be >= 1")
        self.beta_net_: Any = None
        self.factor_net_: Any = None
        self.premia_: np.ndarray = np.zeros(0)
        self.train_report_: Any = None
        self.r2_: dict[str, float] = {}

    def _nets(self, n_char: int) -> tuple[Any, Any]:
        nn = require_torch().nn
        p = self.params
        beta = nn.Sequential(
            nn.Linear(n_char, p["hidden"]), nn.ReLU(), nn.Linear(p["hidden"], p["n_factors"])
        )
        factor = nn.Linear(n_char, p["n_factors"])
        return beta, factor

    def fit(self, data: pd.DataFrame) -> Self:
        p = self.params
        torch = seed_everything(p["seed"])
        close = to_wide(data, "close")
        Z_all = _char_tensor(close, tuple(p["features"]))
        h = p["ret_horizon"]
        fwd = (close.shift(-h) / close - 1.0).to_numpy(dtype=float)  # NaN past the window end
        rows = np.arange(0, len(close), h)
        Z, R = Z_all[rows], fwd[rows]
        mask = ~np.isnan(Z).any(axis=2) & ~np.isnan(R)
        keep = mask.sum(axis=1) >= max(p["min_assets"], Z.shape[2] + 1)
        Z, R, mask = Z[keep], R[keep], mask[keep]
        if len(Z) < 30:
            raise ValueError(f"only {len(Z)} usable training dates; need more history")
        Z, R = np.nan_to_num(Z), np.nan_to_num(R)
        X = managed_portfolios(Z, R, mask)

        n_val = max(5, int(len(Z) * p["val_frac"]))
        tr, va = slice(0, len(Z) - n_val - 1), slice(len(Z) - n_val, len(Z))  # 1-block gap
        T = {k: torch.tensor(v, dtype=torch.float32) for k, v in dict(Z=Z, R=R, X=X).items()}
        M = torch.tensor(mask, dtype=torch.float32)
        beta_net, factor_net = self._nets(Z.shape[2])
        params = list(beta_net.parameters()) + list(factor_net.parameters())

        class Both(torch.nn.Module):  # type: ignore[name-defined]  # one Module to snapshot
            def __init__(self) -> None:
                super().__init__()
                self.beta, self.factor = beta_net, factor_net

        both = Both()

        def recon(sl: slice) -> Any:
            b = beta_net(T["Z"][sl])  # (D, N, K)
            f = factor_net(T["X"][sl])  # (D, K)
            return (b * f[:, None, :]).sum(dim=2)

        def sse(sl: slice) -> tuple[Any, Any]:
            err = (recon(sl) - T["R"][sl]) * M[sl]
            return (err**2).sum(), ((T["R"][sl] * M[sl]) ** 2).sum()

        def step(opt: Any, _train: bool) -> float:
            opt.zero_grad()
            e, tot = sse(tr)
            loss = e / tot + p["l1"] * sum(w.abs().sum() for w in params)
            loss.backward()
            opt.step()
            return float(loss.detach())

        def validate(_m: Any) -> float:
            e, tot = sse(va)
            return float(1.0 - e / tot)  # total R^2 on the validation dates

        self.train_report_ = train_early_stopping(
            both, step, validate, p["max_epochs"], p["patience"], p["lr"], weight_decay=0.0
        )
        both.eval()
        with torch.no_grad():
            f_train = factor_net(T["X"][tr]).numpy()
            e_tr, tot_tr = sse(tr)
            # predictive R^2: beta(z_t) . lambda vs realised returns on validation dates
            lam = torch.tensor(f_train.mean(axis=0), dtype=torch.float32)
            pred_va = (beta_net(T["Z"][va]) * lam).sum(dim=2)
            err = ((pred_va - T["R"][va]) * M[va]) ** 2
            pred_r2 = 1.0 - float(err.sum() / ((T["R"][va] * M[va]) ** 2).sum())
        self.premia_ = f_train.mean(axis=0)
        self.r2_ = {
            "total_train": float(1.0 - e_tr / tot_tr),
            "total_val": self.train_report_.val_score,
            "predictive_val": pred_r2,
        }
        self.beta_net_ = beta_net
        self.factor_net_ = factor_net
        self.is_fitted = True
        return self

    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Expected ``ret_horizon``-day return beta(z_t) . lambda per (date, ticker)."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        torch = require_torch()
        close = to_wide(data, "close")
        Z = _char_tensor(close, tuple(self.params["features"]))
        ok = ~np.isnan(Z).any(axis=2)
        with torch.no_grad():
            b = self.beta_net_(torch.tensor(np.nan_to_num(Z), dtype=torch.float32)).numpy()
        pred = np.where(ok, b @ self.premia_, np.nan)
        return pd.DataFrame(pred, index=close.index, columns=close.columns)

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        return quantile_long_short(
            self.score(data), p["quantile"], p["gross"], p["rebalance_days"], p["min_assets"]
        )
