"""Small PyTorch models for strategies #9 and #10. PyTorch is an optional extra.

Install with ``uv sync --extra dl``. Nothing here imports torch at module load,
so the rest of ``quantskills`` works without it; :func:`require_torch` raises a
clear message when it is missing.

Training is deterministic on CPU (fixed seed, fixed thread count) and uses early
stopping on a **later** slice of the training window (never a random split).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DL_HINT = "this strategy needs PyTorch: run `uv sync --extra dl`"
ARCHS = ("mlp", "cnn", "lstm")


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(DL_HINT) from exc
    return torch


def torch_available() -> bool:
    try:
        require_torch()
    except ImportError:
        return False
    return True


def seed_everything(seed: int, threads: int = 2) -> Any:
    torch = require_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    return torch


def make_sequence_net(arch: str, length: int, hidden: int = 16, dropout: float = 0.1) -> Any:
    """``(batch, length)`` sequence of scaled returns -> one score per sample."""
    torch = require_torch()
    nn = torch.nn
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(length, 2 * hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * hidden, 1)
        )
    if arch == "cnn":

        class CNN(nn.Module):  # type: ignore[name-defined]  # 2 dilated convs + global mean
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, hidden // 2, kernel_size=5),
                    nn.ReLU(),
                    nn.Conv1d(hidden // 2, hidden // 2, kernel_size=5, dilation=2),
                    nn.ReLU(),
                )
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

            def forward(self, x):
                h = self.conv(x.unsqueeze(1)).mean(dim=2)
                return self.head(h)

        return CNN()
    if arch == "lstm":

        class LSTM(nn.Module):  # type: ignore[name-defined]
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

            def forward(self, x):
                out, _ = self.lstm(x.unsqueeze(2))
                return self.head(out[:, -1, :])

        return LSTM()
    raise ValueError(f"arch must be one of {ARCHS}")


@dataclass
class TrainReport:
    best_epoch: int = 0
    epochs_run: int = 0
    val_score: float = float("nan")
    history: list[tuple[float, float]] = field(default_factory=list)  # (train loss, val score)


def train_early_stopping(
    model: Any,
    step: Callable[[Any, bool], float],
    validate: Callable[[Any], float],
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TrainReport:
    """Generic loop. ``step(optimizer, train=True)`` runs one epoch and returns the
    mean training loss; ``validate(model)`` returns a score where **higher is better**.
    Restores the weights from the best validation epoch."""
    torch = require_torch()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rep = TrainReport()
    best = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    since = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss = step(opt, True)
        model.eval()
        with torch.no_grad():
            score = validate(model)
        rep.history.append((float(loss), float(score)))
        rep.epochs_run = epoch
        if np.isfinite(score) and score > best:
            best, rep.best_epoch, since = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    rep.val_score = float(best)
    return rep


def mean_daily_ic(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, min_obs: int = 5) -> float:
    """Mean over groups (dates) of the Spearman correlation of ``pred`` and ``y``."""
    ics = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_obs:
            continue
        a, b = _rank(pred[m]), _rank(y[m])
        if a.std() > 0 and b.std() > 0:
            ics.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(ics)) if ics else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r
