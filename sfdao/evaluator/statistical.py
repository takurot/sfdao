from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats

__all__ = ["KSTestResult", "StatisticalEvaluator"]


@dataclass(frozen=True)
class KSTestResult:
    statistic: float
    p_value: float


class StatisticalEvaluator:
    """Basic statistical similarity checks between real and synthetic datasets."""

    def ks_test(self, real: Iterable[float], synthetic: Iterable[float]) -> KSTestResult:
        real_arr = self._prepare_array(real)
        synthetic_arr = self._prepare_array(synthetic)
        self._ensure_non_empty(real_arr, synthetic_arr)

        statistic, p_value = stats.ks_2samp(real_arr, synthetic_arr)
        return KSTestResult(statistic=float(statistic), p_value=float(p_value))

    def js_divergence(
        self,
        real: Iterable[float],
        synthetic: Iterable[float],
        *,
        bins: int = 50,
    ) -> float:
        real_arr = self._prepare_array(real)
        synthetic_arr = self._prepare_array(synthetic)
        self._ensure_non_empty(real_arr, synthetic_arr)

        value_min = float(min(real_arr.min(), synthetic_arr.min()))
        value_max = float(max(real_arr.max(), synthetic_arr.max()))

        if value_min == value_max:
            return 0.0

        real_hist, bin_edges = np.histogram(real_arr, bins=bins, range=(value_min, value_max), density=True)
        synthetic_hist, _ = np.histogram(
            synthetic_arr, bins=bin_edges, density=True
        )

        real_prob = real_hist + 1e-12
        synthetic_prob = synthetic_hist + 1e-12
        mixture = 0.5 * (real_prob + synthetic_prob)

        divergence = 0.5 * (
            stats.entropy(real_prob, mixture, base=2)
            + stats.entropy(synthetic_prob, mixture, base=2)
        )
        return float(divergence)

    def _prepare_array(self, values: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(values), dtype=float)
        if array.size == 0:
            return array

        if np.isnan(array).any():
            array = array[~np.isnan(array)]
        return array

    def _ensure_non_empty(self, real: np.ndarray, synthetic: np.ndarray) -> None:
        if real.size == 0 or synthetic.size == 0:
            raise ValueError("Real and synthetic arrays must be non-empty.")
