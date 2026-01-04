from __future__ import annotations

from typing import Iterable, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KDTree  # type: ignore

__all__ = ["PrivacyEvaluator"]


class PrivacyEvaluator:
    """Evaluate privacy risks between real and synthetic records."""

    def __init__(self, sample_size: int | None = None) -> None:
        self.sample_size = sample_size

    def distance_to_closest_record(
        self,
        real: Iterable[Iterable[float]] | NDArray[np.float64],
        synthetic: Iterable[Iterable[float]] | NDArray[np.float64],
        progress_callback: Callable[[int], None] | None = None,
        batch_size: int = 1000,
    ) -> NDArray[np.float64]:
        real_arr = self._prepare_matrix(real)
        synthetic_arr = self._prepare_matrix(synthetic)
        self._ensure_non_empty(real_arr, synthetic_arr)

        # Optimization: Use KDTree for nearest neighbor search
        tree = KDTree(real_arr)

        n_synthetic = len(synthetic_arr)
        distances = np.zeros(n_synthetic, dtype=np.float64)

        # Batch processing for progress reporting
        for start_idx in range(0, n_synthetic, batch_size):
            end_idx = min(start_idx + batch_size, n_synthetic)
            batch = synthetic_arr[start_idx:end_idx]

            # k=1 returns (distances, indices) for the nearest neighbor
            batch_dists, _ = tree.query(batch, k=1)
            distances[start_idx:end_idx] = batch_dists.ravel()

            if progress_callback:
                progress_callback(len(batch))

        return distances

    def reidentification_risk(
        self,
        real: Iterable[Iterable[float]] | NDArray[np.float64],
        synthetic: Iterable[Iterable[float]] | NDArray[np.float64],
        progress_callback: Callable[[int], None] | None = None,
    ) -> float:
        real_arr = self._prepare_matrix(real)
        synthetic_arr = self._prepare_matrix(synthetic)
        self._ensure_non_empty(real_arr, synthetic_arr)

        dcr = self.distance_to_closest_record(
            real_arr, synthetic_arr, progress_callback=progress_callback
        )
        scale = self.reference_distance(real_arr)

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

        if self.sample_size is not None and len(array) > self.sample_size:
            indices = np.random.choice(len(array), self.sample_size, replace=False)
            array = array[indices]

        return array.astype(np.float64, copy=False)

    def _ensure_non_empty(self, real: NDArray[np.float64], synthetic: NDArray[np.float64]) -> None:
        if real.size == 0 or synthetic.size == 0:
            raise ValueError("Real and synthetic datasets must be non-empty.")

    def reference_distance(self, real: NDArray[np.float64]) -> float:
        """Calculate median nearest-neighbor distance as a reference scale.

        Uses sampling (default 10k) for large datasets to improve performance.
        """
        if real.shape[0] < 2:
            spread = float(np.max(np.std(real, axis=0, ddof=0))) if real.size else 1.0
            return 1.0 if spread == 0 else spread

        # Optimization: Use KDTree to find nearest neighbor distance (excluding self)
        tree = KDTree(real)

        # Optimization: Sample query points if dataset is large
        # We only need an estimate of the median distance, so a sample is sufficient.
        query_points = real
        # Default to 10k samples for reference distance even if sample_size is None
        # because this is just for scaling parameter estimation.
        ref_sample_limit = self.sample_size if self.sample_size is not None else 10000

        if len(real) > ref_sample_limit:
            indices = np.random.choice(len(real), ref_sample_limit, replace=False)
            query_points = real[indices]

        # k=2 because the nearest neighbor of a point is itself (distance 0)
        distances, _ = tree.query(query_points, k=2)

        # Take the 2nd column (distance to the closest OTHER point)
        nearest_neighbor_dists = distances[:, 1]

        if nearest_neighbor_dists.size == 0:
            return 1.0

        reference = float(np.median(nearest_neighbor_dists))
        if reference == 0.0:
            spread = float(np.max(np.std(real, axis=0, ddof=0)))
            reference = spread if spread > 0 else 1.0
        return reference
