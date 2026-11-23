"""The ``Strategy`` interface every strategy in the library implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import pandas as pd


class Strategy(ABC):
    """A trading strategy.

    * ``fit(data)`` learns anything data-dependent, using **only** ``data``
      (pass the training window, never the hold-out).
    * ``signals(data)`` returns target weights, wide ``date x ticker``, as
      fractions of NAV. Row *t* may use data up to and including *t*. An
      all-NaN row means "hold / don't rebalance".
    * ``params`` holds every tunable setting, so runs are reproducible.

    ``data`` is the long ``MultiIndex(date, ticker)`` OHLCV frame.
    """

    name: ClassVar[str] = "base"

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.default_params())
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown params {sorted(unknown)}")
        self.params: dict[str, Any] = {**self.default_params(), **params}
        self.is_fitted = False

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> Self: ...

    @abstractmethod
    def signals(self, data: pd.DataFrame) -> pd.DataFrame: ...

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"
