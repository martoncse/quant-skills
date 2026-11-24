"""A compact TimeGAN for multivariate return windows (strategy #11). Needs ``--extra dl``.

After Yoon, Jarrett & van der Schaar (2019), "Time-series Generative Adversarial
Networks", written from scratch and kept small. Five GRU networks:

* embedder  E: data x (L x d)        -> latent h (L x hidden)
* recovery  R: latent h               -> data x~        (autoencoder with E)
* generator G: noise z                -> latent h^
* supervisor S: latent h_t            -> h_{t+1}         (teaches step-to-step dynamics)
* discriminator D: latent sequence    -> real / fake score per step

Training has three phases: (1) the autoencoder E/R; (2) the supervisor on real
latents; (3) joint adversarial training, where G is also pushed to match the
supervisor's dynamics and the data's moments.

Scaling: the paper min-max scales each feature to [0, 1]. With fat-tailed returns a
few extreme days set the range, ordinary days collapse into a sliver around 0.5, and
so does the generator. By default we standardise and squash instead,
u = 0.5 + 0.5 tanh(z / 2), which keeps ordinary days spread out and inverts exactly.

Deviations from the paper, for a laptop CPU budget (hundreds of steps, not tens of
thousands): single-layer GRUs by default, tanh scaling (above), and the moment loss
also matches the cross-feature covariance and lag-1 autocorrelation (``_moment_gap``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantskills.models.nn import require_torch, seed_everything


def make_windows(returns: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    """``(T, d)`` -> ``(n, seq_len, d)`` overlapping windows (rows with NaN dropped first)."""
    r = returns[np.isfinite(returns).all(axis=1)]
    if len(r) < seq_len:
        raise ValueError("fewer rows than seq_len")
    idx = np.arange(0, len(r) - seq_len + 1, stride)
    return np.stack([r[i : i + seq_len] for i in idx])


def _moment_gap(x_hat: Any, x: Any) -> Any:
    """Distance between batch moments: per-step mean and std (as in TimeGAN), plus the
    cross-feature covariance and lag-1 autocorrelation (our addition: at small training
    budgets the generator otherwise couples features and over-smooths)."""
    gap = (x_hat.std(0) - x.std(0)).abs().mean() + (x_hat.mean(0) - x.mean(0)).abs().mean()

    def cov(z: Any) -> Any:
        f = z.reshape(-1, z.shape[-1])
        f = f - f.mean(0)
        return f.T @ f / len(f)

    def ac1(z: Any) -> Any:
        c = z - z.mean(dim=(0, 1))
        return (c[:, 1:] * c[:, :-1]).mean(dim=(0, 1)) / (c.pow(2).mean(dim=(0, 1)) + 1e-8)

    return gap + 10 * (cov(x_hat) - cov(x)).abs().mean() + (ac1(x_hat) - ac1(x)).abs().mean()


@dataclass
class TimeGANConfig:
    seq_len: int = 24
    hidden: int = 24
    layers: int = 1
    noise_dim: int | None = None  # default: number of features
    batch_size: int = 64
    lr: float = 1e-3
    embed_steps: int = 600
    supervised_steps: int = 600
    joint_steps: int = 600
    gamma: float = 1.0  # weight of the "unsupervised" adversarial term
    seed: int = 0
    threads: int = 4
    scaling: str = "tanh"  # "tanh" (default, robust to fat tails) or "minmax" (paper)


@dataclass
class TimeGANLog:
    embed: list[float] = field(default_factory=list)
    supervised: list[float] = field(default_factory=list)
    generator: list[float] = field(default_factory=list)
    discriminator: list[float] = field(default_factory=list)


class TimeGAN:
    def __init__(self, cfg: TimeGANConfig | None = None) -> None:
        self.cfg = cfg or TimeGANConfig()
        self.lo_: np.ndarray = np.zeros(0)
        self.hi_: np.ndarray = np.zeros(0)
        self.nets_: dict[str, Any] = {}
        self.log_ = TimeGANLog()
        self.n_features_ = 0

    def _build(self, d: int) -> None:
        torch = require_torch()
        nn = torch.nn
        c = self.cfg
        h, z = c.hidden, c.noise_dim or d

        class Block(nn.Module):  # type: ignore[name-defined]
            def __init__(self, n_in: int, n_out: int, layers: int, act: bool = True) -> None:
                super().__init__()
                self.rnn = nn.GRU(n_in, h, num_layers=layers, batch_first=True)
                self.out = nn.Linear(h, n_out)
                self.act = act

            def forward(self, x):
                y = self.out(self.rnn(x)[0])
                return torch.sigmoid(y) if self.act else y

        self.nets_ = {
            "E": Block(d, h, c.layers),
            "R": Block(h, d, c.layers),
            "G": Block(z, h, c.layers),
            "S": Block(h, h, max(1, c.layers - 1)),
            "D": Block(h, 1, c.layers, act=False),
        }
        self.noise_dim_ = z

    def _noise(self, n: int, gen: Any) -> Any:
        torch = require_torch()
        return torch.rand(n, self.cfg.seq_len, self.noise_dim_, generator=gen)

    def fit(self, windows: np.ndarray) -> TimeGAN:
        """``windows``: ``(n, seq_len, d)`` real data, e.g. from :func:`make_windows`."""
        torch = seed_everything(self.cfg.seed, threads=self.cfg.threads)
        c = self.cfg
        if windows.ndim != 3 or windows.shape[1] != c.seq_len:
            raise ValueError(f"windows must be (n, {c.seq_len}, d)")
        n, _, d = windows.shape
        self.n_features_ = d
        if c.scaling == "minmax":
            self.lo_ = windows.min(axis=(0, 1))
            self.hi_ = windows.max(axis=(0, 1))
        elif c.scaling == "tanh":
            self.lo_ = windows.mean(axis=(0, 1))  # centre
            self.hi_ = windows.std(axis=(0, 1))  # scale
        else:
            raise ValueError("scaling must be 'tanh' or 'minmax'")
        x_all = torch.tensor(self._to_unit(windows), dtype=torch.float32)
        self._build(d)
        E, R, G, S, D = (self.nets_[k] for k in "ERGSD")
        gen = torch.Generator().manual_seed(c.seed)
        mse, bce = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()

        def batch() -> Any:
            return x_all[torch.randint(0, n, (c.batch_size,), generator=gen)]

        def opt(*nets: Any) -> Any:
            return torch.optim.Adam([p for m in nets for p in m.parameters()], lr=c.lr)

        # Phase 1: autoencoder.
        o_er = opt(E, R)
        for _ in range(c.embed_steps):
            x = batch()
            loss = mse(R(E(x)), x)
            o_er.zero_grad()
            (10 * torch.sqrt(loss)).backward()
            o_er.step()
            self.log_.embed.append(float(loss.detach()))

        # Phase 2: supervisor learns one-step latent dynamics.
        o_s = opt(S)
        for _ in range(c.supervised_steps):
            h = E(batch()).detach()
            loss = mse(S(h)[:, :-1], h[:, 1:])
            o_s.zero_grad()
            loss.backward()
            o_s.step()
            self.log_.supervised.append(float(loss.detach()))

        # Phase 3: joint training.
        o_g, o_e, o_d = opt(G, S), opt(E, R), opt(D)
        ones = torch.ones(c.batch_size, c.seq_len, 1)
        zeros = torch.zeros_like(ones)
        for _ in range(c.joint_steps):
            for _k in range(2):  # generator twice per discriminator step
                x = batch()
                e_hat = G(self._noise(c.batch_size, gen))
                h_hat = S(e_hat)
                x_hat = R(h_hat)
                h = E(x)
                g_adv = bce(D(h_hat), ones) + c.gamma * bce(D(e_hat), ones)
                g_sup = mse(S(h)[:, :-1], h[:, 1:])
                g_mom = _moment_gap(x_hat, x)
                g_loss = g_adv + 100 * torch.sqrt(g_sup) + 100 * g_mom
                o_g.zero_grad()
                g_loss.backward()
                o_g.step()
                # keep the embedding faithful while the generator moves
                h = E(x)
                e_loss = 10 * torch.sqrt(mse(R(h), x)) + 0.1 * mse(S(h)[:, :-1], h[:, 1:])
                o_e.zero_grad()
                e_loss.backward()
                o_e.step()
            x = batch()
            e_hat = G(self._noise(c.batch_size, gen)).detach()
            h_hat = S(e_hat).detach()
            d_loss = (
                bce(D(E(x).detach()), ones) + bce(D(h_hat), zeros) + c.gamma * bce(D(e_hat), zeros)
            )
            if float(d_loss.detach()) > 0.15:  # don't let D get too far ahead
                o_d.zero_grad()
                d_loss.backward()
                o_d.step()
            self.log_.generator.append(float(g_loss.detach()))
            self.log_.discriminator.append(float(d_loss.detach()))
        for m in self.nets_.values():
            m.eval()
        return self

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """``(n, seq_len, d)`` synthetic windows in the original units."""
        if not self.nets_:
            raise RuntimeError("call fit() first")
        torch = require_torch()
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            x = self.nets_["R"](self.nets_["S"](self.nets_["G"](self._noise(n, gen)))).numpy()
        return self._from_unit(x)

    _K = 2.0  # tanh squash width, in standard deviations

    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.scaling == "minmax":
            return (x - self.lo_) / (self.hi_ - self.lo_ + 1e-12)
        return 0.5 + 0.5 * np.tanh((x - self.lo_) / (self._K * self.hi_ + 1e-12))

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        if self.cfg.scaling == "minmax":
            return u * (self.hi_ - self.lo_ + 1e-12) + self.lo_
        z = np.arctanh(np.clip(2.0 * u - 1.0, -0.999, 0.999))
        return self.lo_ + self._K * self.hi_ * z
