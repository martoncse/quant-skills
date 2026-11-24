"""Strategy #12: a deep reinforcement-learning trader (Double DQN) on one asset (dl extra).

``fit`` trains the agent in the trading environment on the first ``1 - val_frac``
of the training window and keeps the checkpoint with the best Sharpe on the last
``val_frac`` (a later slice, never a random one). ``signals`` runs the frozen greedy
policy day by day: the position it holds is part of what it sees, so it learns
when trading is worth the cost.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from quantskills.data.loader import to_wide
from quantskills.models.rl import (
    DQNConfig,
    greedy_positions,
    make_env,
    scaled_return_features,
    train_dqn,
    trend_state_features,
)
from quantskills.strategies.base import Strategy


class RLAgent(Strategy):
    """#12: Double-DQN agent choosing short / flat / long on one asset (dl extra)."""

    name = "rl-dqn"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "ticker": "",  # empty: SPY if present, else the first ticker
            "state": "trend",  # "trend": returns over `lookbacks`; "raw": last `window` returns
            "lookbacks": (1, 5, 20, 60),
            "window": 10,
            "vol_window": 63,
            "cost_bps": 10.0,  # charged in the reward, matching the backtester's default
            "hold_days": 5,  # decide once a week; daily decisions mostly chase noise
            "episode_len": 252,
            "train_steps": 20_000,
            "gamma": 0.9,
            "hidden": 64,
            "lr": 1e-3,
            "eval_every": 2_000,
            "val_frac": 0.2,
            "seed": 0,
        }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        if self.params["state"] not in ("trend", "raw"):
            raise ValueError("state must be 'trend' or 'raw'")
        if not 0 < self.params["val_frac"] < 0.5:
            raise ValueError("val_frac must be in (0, 0.5)")
        self.ticker_ = ""
        self.net_: Any = None
        self.report_: Any = None

    def _pick(self, close: pd.DataFrame) -> str:
        want = str(self.params["ticker"]).upper()
        if want:
            if want not in close.columns:
                raise KeyError(f"ticker {want!r} not in data")
            return want
        return "SPY" if "SPY" in close.columns else str(sorted(close.columns)[0])

    def _features(self, close: pd.Series) -> np.ndarray:
        p = self.params
        if p["state"] == "trend":
            return trend_state_features(close, tuple(p["lookbacks"]), p["vol_window"])
        return scaled_return_features(close, p["window"], p["vol_window"])

    def fit(self, data: pd.DataFrame) -> Self:
        p = self.params
        close = to_wide(data, "close")
        self.ticker_ = self._pick(close)
        px = close[self.ticker_]
        X = self._features(px)
        r = px.pct_change(fill_method=None).to_numpy(dtype=float)
        cut = int(len(px) * (1 - p["val_frac"]))
        # Validation starts after the training slice; its features look back into it,
        # which is fine (past data), but no training reward uses a validation return.
        env = make_env(
            X[:cut],
            r[:cut],
            cost_bps=p["cost_bps"],
            episode_len=p["episode_len"],
            seed=p["seed"],
            hold_days=p["hold_days"],
        )
        cfg = DQNConfig(
            hidden=p["hidden"],
            lr=p["lr"],
            gamma=p["gamma"],
            train_steps=p["train_steps"],
            eval_every=p["eval_every"],
            learning_starts=min(500, p["train_steps"] // 2),
            seed=p["seed"],
        )
        self.net_, self.report_ = train_dqn(
            env, X[cut:], r[cut:], cfg, p["cost_bps"], p["hold_days"]
        )
        self.is_fitted = True
        return self

    def positions(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("call fit() before signals()")
        px = to_wide(data, "close")[self.ticker_]
        pos = greedy_positions(self.net_, self._features(px), self.params["hold_days"])
        return pd.Series(pos, index=px.index)

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = to_wide(data, "close")
        w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        w[self.ticker_] = self.positions(data)
        return w
