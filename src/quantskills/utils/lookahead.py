"""A reusable look-ahead test.

Idea: if a feature at time *t* only uses data up to *t*, then scrambling every
row after *t* must leave the feature's values at rows <= *t* unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def assert_no_lookahead(
    fn: Callable[[pd.DataFrame], pd.DataFrame],
    data: pd.DataFrame,
    cut_points: Sequence[int] | None = None,
    seed: int = 0,
) -> None:
    """Raise ``AssertionError`` if ``fn``'s output at row t depends on rows > t.

    ``data`` is a wide ``date x ticker`` frame (positive values, e.g. prices).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    cut_points = cut_points if cut_points is not None else [n // 3, n // 2, (3 * n) // 4]
    base = fn(data)
    for cut in cut_points:
        scrambled = data.copy()
        noise = rng.lognormal(0.0, 0.3, size=scrambled.iloc[cut + 1 :].shape)
        scrambled.iloc[cut + 1 :] = scrambled.iloc[cut + 1 :].to_numpy() * noise
        out = fn(scrambled)
        a = base.iloc[: cut + 1]
        b = out.iloc[: cut + 1]
        pd.testing.assert_frame_equal(
            a, b, check_exact=False, rtol=1e-10, obj=f"rows <= {cut} after scrambling the future"
        )
