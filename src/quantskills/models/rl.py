"""A single-asset trading environment and a Double-DQN agent (strategy #12). Needs ``--extra dl``.

Environment (gym-style, ``gymnasium.Env`` via :func:`make_env`):
* **observation** at day *t*: the last ``window`` daily returns divided by trailing
  volatility (past data only), plus the position currently held (-1, 0 or +1);
* **action**: 0 = short, 1 = flat, 2 = long, traded at the close of *t* and held for
  ``hold_days`` days;
* **reward**: position x return over the holding period, minus ``cost_bps`` per unit
  traded, in percent. Same timing and cost convention as the backtester.

Agent: Double DQN (van Hasselt et al., 2016) with a replay buffer, a target network,
Huber loss and linearly decaying epsilon-greedy exploration. Written from scratch;
no RL library is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantskills.models.nn import require_torch, seed_everything

ACTIONS = np.array([-1.0, 0.0, 1.0])  # short, flat, long


def scaled_return_features(close: pd.Series, window: int = 10, vol_window: int = 63) -> np.ndarray:
    """``(T, window)``: row t holds returns t-window+1..t over trailing vol at t. NaN in warm-up."""
    r = close.pct_change(fill_method=None)
    vol = r.rolling(vol_window, min_periods=vol_window // 2).std()
    z = (r / vol).clip(-5, 5).to_numpy(dtype=float)
    out = np.full((len(z), window), np.nan)
    for k in range(window):
        out[k:, window - 1 - k] = z[: len(z) - k]
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out


def trend_state_features(
    close: pd.Series, lookbacks: tuple[int, ...] = (1, 5, 20, 60), vol_window: int = 63
) -> np.ndarray:
    """``(T, len(lookbacks))``: return over each lookback, in units of its expected std.

    z_h(t) = (P_t / P_{t-h} - 1) / (sigma_t * sqrt(h)), sigma from the trailing
    ``vol_window`` days. A compact state that is hard to memorise. Past data only.
    """
    r = close.pct_change(fill_method=None)
    vol = r.rolling(vol_window, min_periods=vol_window // 2).std()
    cols = [((close / close.shift(h) - 1.0) / (vol * np.sqrt(h))).clip(-5, 5) for h in lookbacks]
    out = np.column_stack([c.to_numpy(dtype=float) for c in cols])
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out


class TradingCore:
    """The environment's maths, without gymnasium (easy to test)."""

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        cost_bps: float = 10.0,
        episode_len: int = 252,
        seed: int = 0,
        hold_days: int = 1,
    ) -> None:
        if hold_days < 1:
            raise ValueError("hold_days must be >= 1")
        self.hold = hold_days
        ok = np.isfinite(features).all(axis=1) & np.isfinite(np.r_[returns[1:], np.nan])
        self.valid = np.flatnonzero(ok)  # days t with features and a next-day return
        if len(self.valid) < episode_len + 1:
            raise ValueError("not enough data for one episode")
        self.X, self.r = features, returns
        self.cost = cost_bps / 1e4
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self.t = self.end = 0
        self.pos = 0.0

    def obs(self) -> np.ndarray:
        return np.append(self.X[self.t], self.pos).astype(np.float32)

    def reset(self, start: int | None = None) -> np.ndarray:
        hi = len(self.valid) - self.episode_len
        i = int(self.rng.integers(0, hi)) if start is None else start
        self.t, self.end = int(self.valid[i]), int(self.valid[i + self.episode_len - 1])
        self.pos = 0.0
        return self.obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        new = float(ACTIONS[action])
        k = min(self.hold, len(self.r) - 1 - self.t)  # hold for k days, then decide again
        held = np.nansum(self.r[self.t + 1 : self.t + 1 + k])
        reward = 100.0 * (new * held - self.cost * abs(new - self.pos))
        self.pos = new
        self.t += k
        done = self.t + self.hold > self.end
        return self.obs(), float(reward), done


def make_env(features: np.ndarray, returns: np.ndarray, **kwargs: Any) -> Any:
    """Wrap :class:`TradingCore` as a ``gymnasium.Env`` (Discrete(3) actions)."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the RL environment needs gymnasium: `uv sync --extra dl`") from exc

    class TradingEnv(gym.Env):  # type: ignore[misc]
        metadata: dict[str, Any] = {"render_modes": []}

        def __init__(self) -> None:
            self.core = TradingCore(features, returns, **kwargs)
            dim = features.shape[1] + 1
            self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
            self.action_space = spaces.Discrete(3)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            if seed is not None:
                self.core.rng = np.random.default_rng(seed)
            return self.core.reset(), {}

        def step(self, action: int):
            obs, reward, done = self.core.step(int(action))
            return obs, reward, done, False, {}

    return TradingEnv()


@dataclass
class DQNConfig:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.9
    batch_size: int = 64
    buffer_size: int = 20_000
    train_steps: int = 20_000
    learning_starts: int = 500
    target_update: int = 250
    eps_start: float = 1.0
    eps_final: float = 0.05
    eps_fraction: float = 0.6
    eval_every: int = 2_000
    seed: int = 0


@dataclass
class DQNReport:
    checkpoints: list[tuple[int, float]] = field(default_factory=list)  # (step, val Sharpe)
    best_step: int = 0
    best_val_sharpe: float = float("nan")


def q_network(n_inputs: int, hidden: int) -> Any:
    nn = require_torch().nn
    return nn.Sequential(
        nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 3),
    )  # fmt: skip


def greedy_positions(net: Any, features: np.ndarray, hold_days: int = 1) -> np.ndarray:
    """Run the greedy policy day by day; the position feeds back into the next observation.

    Q-values are computed for every (day, previous position) at once, then the
    path is followed in numpy. NaN features -> flat.
    """
    torch = require_torch()
    t_len = len(features)
    x = np.nan_to_num(features)
    batch = np.concatenate([np.repeat(x, 3, axis=0), np.tile(ACTIONS, t_len)[:, None]], axis=1)
    with torch.no_grad():
        q = net(torch.tensor(batch, dtype=torch.float32)).numpy().reshape(t_len, 3, 3)
    ok = np.isfinite(features).all(axis=1)
    pos = np.zeros(t_len)
    prev = 1  # index of "flat" in ACTIONS
    since = hold_days  # decide on the first valid day, then every `hold_days` days
    for t in range(t_len):
        if not ok[t]:
            a, since = 1, hold_days
        elif since >= hold_days:
            a, since = int(q[t, prev].argmax()), 0
        else:
            a = prev
        pos[t], prev = ACTIONS[a], a
        since += 1
    return pos


def policy_sharpe(
    net: Any, features: np.ndarray, returns: np.ndarray, cost_bps: float, hold_days: int = 1
) -> float:
    pos = greedy_positions(net, features, hold_days)
    pnl = pos[:-1] * returns[1:] - cost_bps / 1e4 * np.abs(np.diff(np.r_[0.0, pos]))[:-1]
    pnl = pnl[np.isfinite(pnl)]
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    return float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def train_dqn(
    env: Any,
    val_features: np.ndarray,
    val_returns: np.ndarray,
    cfg: DQNConfig,
    cost_bps: float,
    hold_days: int = 1,
) -> tuple[Any, DQNReport]:
    """Double DQN on ``env``; keeps the checkpoint with the best validation Sharpe."""
    torch = seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_in = env.observation_space.shape[0]
    net, target = q_network(n_in, cfg.hidden), q_network(n_in, cfg.hidden)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.SmoothL1Loss()

    obs_buf = np.zeros((cfg.buffer_size, n_in), np.float32)
    nxt_buf = np.zeros((cfg.buffer_size, n_in), np.float32)
    act_buf = np.zeros(cfg.buffer_size, np.int64)
    rew_buf = np.zeros(cfg.buffer_size, np.float32)
    done_buf = np.zeros(cfg.buffer_size, np.float32)
    size = ptr = 0

    report = DQNReport()
    best = -np.inf
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    obs, _ = env.reset(seed=cfg.seed)
    for step in range(1, cfg.train_steps + 1):
        frac = min(1.0, step / (cfg.eps_fraction * cfg.train_steps))
        eps = cfg.eps_start + frac * (cfg.eps_final - cfg.eps_start)
        if rng.uniform() < eps:
            action = int(rng.integers(3))
        else:
            with torch.no_grad():
                action = int(net(torch.tensor(obs[None])).argmax())
        nxt, reward, done, _, _ = env.step(action)
        obs_buf[ptr], nxt_buf[ptr], act_buf[ptr] = obs, nxt, action
        rew_buf[ptr], done_buf[ptr] = reward, float(done)
        ptr, size = (ptr + 1) % cfg.buffer_size, min(size + 1, cfg.buffer_size)
        obs = env.reset()[0] if done else nxt

        if step >= cfg.learning_starts:
            idx = rng.integers(0, size, cfg.batch_size)
            o, n = torch.tensor(obs_buf[idx]), torch.tensor(nxt_buf[idx])
            a, r = torch.tensor(act_buf[idx]), torch.tensor(rew_buf[idx])
            d = torch.tensor(done_buf[idx])
            with torch.no_grad():  # Double DQN: online net picks, target net evaluates
                best_next = net(n).argmax(dim=1, keepdim=True)
                y = r + cfg.gamma * (1 - d) * target(n).gather(1, best_next).squeeze(1)
            q = net(o).gather(1, a[:, None]).squeeze(1)
            loss = loss_fn(q, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % cfg.target_update == 0:
            target.load_state_dict(net.state_dict())
        if step % cfg.eval_every == 0 or step == cfg.train_steps:
            sr = policy_sharpe(net, val_features, val_returns, cost_bps, hold_days)
            report.checkpoints.append((step, sr))
            if sr > best:
                best, report.best_step = sr, step
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    report.best_val_sharpe = float(best)
    return net.eval(), report
