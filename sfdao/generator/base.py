from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

__all__ = ["BaseGenerator"]


class BaseGenerator(ABC):
    """Base interface for Phase 2 generators."""

    def __init__(self, *, seed: int | None = None, **kwargs: Any) -> None:
        self._seed = seed
        self.params = kwargs

    @property
    def seed(self) -> int | None:
        return self._seed

    @abstractmethod
    def fit(self, real: pd.DataFrame) -> None:
        """Fit a generator to the real dataset."""

    @abstractmethod
    def sample(self, n_samples: int) -> pd.DataFrame:
        """Sample synthetic records after fitting."""
