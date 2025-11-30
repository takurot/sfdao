from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = ["PrivacyEvaluator"]


class PrivacyEvaluator:
    """Evaluate privacy risks between real and synthetic records."""

    def distance_to_closest_record(
        self,
        real: Iterable[Iterable[float]] | NDArray[np.float64],
        synthetic: Iterable[Iterable[float]] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        real_arr = self._prepare_matrix(real)
        synthetic_arr = self._prepare_matrix(synthetic)
        self._ensure_non_empty(real_arr, synthetic_arr)

        distances = np.empty(synthetic_arr.shape[0], dtype=np.float64)
        for idx, record in enumerate(synthetic_arr):
            diffs = real_arr - record
            norms = np.linalg.norm(diffs, axis=1)
            distances[idx] = float(np.min(norms))
        return distances

    def reidentification_risk(
        self,
        real: Iterable[Iterable[float]] | NDArray[np.float64],
        synthetic: Iterable[Iterable[float]] | NDArray[np.float64],
    ) -> float:
        real_arr = self._prepare_matrix(real)
        synthetic_arr = self._prepare_matrix(synthetic)
        self._ensure_non_empty(real_arr, synthetic_arr)

        dcr = self.distance_to_closest_record(real_arr, synthetic_arr)
        scale = self._reference_distance(real_arr)

        scaled = np.exp(-dcr / (scale + 1e-12))
        clipped = np.clip(scaled, 0.0, 1.0)
        return float(np.mean(clipped))

    def _prepare_matrix(
        self, values: Iterable[Iterable[float]] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return np.empty((0, 1), dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        if array.size == 0:
            return np.empty((0, array.shape[1]), dtype=np.float64)

        if np.isnan(array).any():
            row_mask = ~np.isnan(array).any(axis=1)
            array = array[row_mask]
        return array.astype(np.float64, copy=False)

    def _ensure_non_empty(self, real: NDArray[np.float64], synthetic: NDArray[np.float64]) -> None:
        if real.size == 0 or synthetic.size == 0:
            raise ValueError("Real and synthetic datasets must be non-empty.")

    def _reference_distance(self, real: NDArray[np.float64]) -> float:
        if real.shape[0] < 2:
            spread = float(np.max(np.std(real, axis=0, ddof=0))) if real.size else 1.0
            return 1.0 if spread == 0 else spread

        nearest_neighbors = []
        for idx in range(real.shape[0]):
            others = np.concatenate((real[:idx], real[idx + 1 :]), axis=0)
            if others.size == 0:
                continue
            distances = np.linalg.norm(others - real[idx], axis=1)
            nearest_neighbors.append(float(np.min(distances)))

        if not nearest_neighbors:
            return 1.0

        reference = float(np.median(nearest_neighbors))
        if reference == 0.0:
            spread = float(np.max(np.std(real, axis=0, ddof=0)))
            reference = spread if spread > 0 else 1.0
        return reference
