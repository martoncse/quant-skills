"""Calibration of probabilistic forecasts (e.g. P(stock beats the median next month)).

A forecast is *calibrated* if, among all the times it said 70%, the event happened
about 70% of the time. Ranking strategies only need the order to be right, but
anything that sizes positions or reports confidence from a probability needs it
to be calibrated too.

* Brier score  = mean((p - y)^2). 0 is perfect; always saying 0.5 scores 0.25.
* Murphy decomposition (binned): Brier ~ reliability - resolution + uncertainty.
  *Reliability* (lower is better) is the calibration error; *resolution* (higher
  is better) is how much the forecasts separate outcomes from the base rate.
* ECE = sum over bins of (bin share) * |mean forecast - hit rate|.

References: Brier (1950); Murphy (1973); Guo et al. (2017), "On Calibration of
Modern Neural Networks"; Niculescu-Mizil & Caruana (2005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.shape != y_arr.shape:
        raise ValueError("p and y must have the same shape")
    ok = ~(np.isnan(p_arr) | np.isnan(y_arr))
    p_arr, y_arr = p_arr[ok], y_arr[ok]
    if len(p_arr) == 0:
        raise ValueError("no complete (p, y) pairs")
    if np.any((p_arr < 0) | (p_arr > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(y_arr, (0.0, 1.0)).all():
        raise ValueError("outcomes must be 0 or 1")
    return p_arr, y_arr


def brier_score(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    p_arr, y_arr = _clean(p, y)
    return float(np.mean((p_arr - y_arr) ** 2))


def log_loss(p: np.ndarray | pd.Series, y: np.ndarray | pd.Series, eps: float = 1e-12) -> float:
    p_arr, y_arr = _clean(p, y)
    q = np.clip(p_arr, eps, 1 - eps)
    return float(-np.mean(y_arr * np.log(q) + (1 - y_arr) * np.log(1 - q)))


def reliability_table(
    p: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """One row per non-empty bin: count, mean forecast, observed hit rate.

    ``strategy="quantile"`` puts equal counts in each bin, which suits forecasts
    bunched near 0.5 (typical in finance); ``"uniform"`` uses equal-width bins.
    """
    p_arr, y_arr = _clean(p, y)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p_arr, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    if len(edges) < 2:  # every forecast identical: one bin
        edges = np.array([edges[0], edges[0]])
    idx = np.clip(np.searchsorted(edges, p_arr, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "p": p_arr, "y": y_arr})
    g = df.groupby("bin")
    out = pd.DataFrame(
        {
            "count": g.size(),
            "mean_pred": g["p"].mean(),
            "hit_rate": g["y"].mean(),
            "lo": edges[g.size().index],
            "hi": edges[g.size().index + 1],
        }
    )
    out["gap"] = out["hit_rate"] - out["mean_pred"]
    return out.reset_index(drop=True)


def expected_calibration_error(
    p: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    t = reliability_table(p, y, n_bins, strategy)
    w = t["count"] / t["count"].sum()
    return float((w * t["gap"].abs()).sum())


def brier_decomposition(
    p: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict[str, float]:
    """Murphy decomposition. Exact only when forecasts are constant within bins."""
    p_arr, y_arr = _clean(p, y)
    t = reliability_table(p_arr, y_arr, n_bins, strategy)
    w = t["count"] / t["count"].sum()
    base = float(y_arr.mean())
    return {
        "brier": brier_score(p_arr, y_arr),
        "reliability": float((w * (t["mean_pred"] - t["hit_rate"]) ** 2).sum()),
        "resolution": float((w * (t["hit_rate"] - base) ** 2).sum()),
        "uncertainty": base * (1.0 - base),
        "base_rate": base,
    }


def calibration_summary(
    p: np.ndarray | pd.Series, y: np.ndarray | pd.Series, n_bins: int = 10
) -> dict[str, float]:
    """Brier, log loss, ECE and the decomposition, plus the climatology Brier to beat."""
    d = brier_decomposition(p, y, n_bins)
    return {
        **d,
        "log_loss": log_loss(p, y),
        "ece": expected_calibration_error(p, y, n_bins),
        "brier_climatology": d["uncertainty"],  # always forecasting the base rate
        "brier_skill": 1.0 - d["brier"] / d["uncertainty"] if d["uncertainty"] > 0 else np.nan,
    }
