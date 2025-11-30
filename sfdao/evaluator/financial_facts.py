from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore[import-untyped]

__all__ = [
    "FatTailResult",
    "VolatilityClusteringResult",
    "FinancialFactsChecker",
]


@dataclass(frozen=True)
class FatTailResult:
    kurtosis: float
    excess_kurtosis: float
    sample_size: int


@dataclass(frozen=True)
class VolatilityClusteringResult:
    ljung_box_statistic: float
    ljung_box_p_value: float
    arch_test_statistic: float
    arch_test_p_value: float
    lags: int


class FinancialFactsChecker:
    """Evaluate financial stylized facts such as fat tails and volatility clustering."""

    def check_fat_tail(self, data: Iterable[float]) -> FatTailResult:
        values = self._prepare_array(data)
        self._ensure_non_empty(values)

        kurtosis_value = float(stats.kurtosis(values, fisher=False, bias=False))
        excess_kurtosis = kurtosis_value - 3.0

        return FatTailResult(
            kurtosis=kurtosis_value,
            excess_kurtosis=excess_kurtosis,
            sample_size=values.size,
        )

    def check_volatility_clustering(
        self,
        returns: Iterable[float],
        *,
        lags: int = 10,
    ) -> VolatilityClusteringResult:
        values = self._prepare_array(returns)
        self._ensure_min_length(values, lags + 1)

        centered = values - values.mean()
        squared = centered**2

        ljung_box_stat, ljung_box_p = self._ljung_box_test(squared, lags)
        arch_stat, arch_p = self._arch_lm_test(squared, lags)

        return VolatilityClusteringResult(
            ljung_box_statistic=ljung_box_stat,
            ljung_box_p_value=ljung_box_p,
            arch_test_statistic=arch_stat,
            arch_test_p_value=arch_p,
            lags=lags,
        )

    def _ljung_box_test(self, data: NDArray[np.float64], lags: int) -> tuple[float, float]:
        n = data.size
        autocorrs = self._autocorrelations(data, lags)
        weights = (n - np.arange(1, lags + 1)).astype(np.float64)
        q_stat = float(n * (n + 2) * np.sum((autocorrs**2) / weights))
        p_value = float(1 - stats.chi2.cdf(q_stat, df=lags))
        return q_stat, p_value

    def _arch_lm_test(self, data: NDArray[np.float64], lags: int) -> tuple[float, float]:
        n = data.size
        y = data[lags:]
        if y.size == 0:
            raise ValueError("Not enough observations for ARCH LM test.")

        x_columns = [np.ones_like(y)]
        for lag in range(1, lags + 1):
            x_columns.append(data[lags - lag : n - lag])

        x_matrix = np.column_stack(x_columns)
        coefficients, *_ = np.linalg.lstsq(x_matrix, y, rcond=None)
        residuals = y - x_matrix @ coefficients

        ss_total = float(np.sum((y - y.mean()) ** 2))
        ss_resid = float(np.sum(residuals**2))
        r_squared = 0.0 if ss_total == 0 else 1.0 - ss_resid / ss_total

        lm_stat = float(y.size * r_squared)
        p_value = float(1 - stats.chi2.cdf(lm_stat, df=lags))
        return lm_stat, p_value

    def _autocorrelations(self, data: NDArray[np.float64], lags: int) -> NDArray[np.float64]:
        mean = data.mean()
        centered = data - mean
        denominator = float(np.sum(centered**2))
        if denominator == 0:
            return np.zeros(lags, dtype=np.float64)

        autocorrs = np.empty(lags, dtype=np.float64)
        for lag in range(1, lags + 1):
            numerator = float(np.sum(centered[lag:] * centered[:-lag]))
            autocorrs[lag - 1] = numerator / denominator
        return autocorrs

    def _prepare_array(self, values: Iterable[float]) -> NDArray[np.float64]:
        array = np.asarray(list(values), dtype=float)
        if array.size == 0:
            return array.astype(np.float64)

        if np.isnan(array).any():
            array = array[~np.isnan(array)]
        return array.astype(np.float64)

    def _ensure_non_empty(self, values: NDArray[np.float64]) -> None:
        if values.size == 0:
            raise ValueError("Input array must be non-empty.")

    def _ensure_min_length(self, values: NDArray[np.float64], minimum: int) -> None:
        if values.size < minimum:
            raise ValueError(f"Input array must have at least {minimum} observations.")
