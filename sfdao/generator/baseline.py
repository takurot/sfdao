from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sfdao.generator.base import BaseGenerator
from sfdao.ingestion.type_detector import ColumnType, TypeDetector
from sfdao.guard.engine import GuardEngine
from sfdao.scenario.engine import ScenarioEngine

__all__ = ["BaselineGenerator"]


@dataclass(frozen=True)
class _NumericModel:
    model_type: Literal["numeric"]
    missing_rate: float
    mean: float
    std: float
    constant_value: float | None


@dataclass(frozen=True)
class _DiscreteModel:
    model_type: Literal["discrete"]
    missing_rate: float
    choices: list[object]
    probabilities: list[float]


@dataclass(frozen=True)
class _AllMissingModel:
    model_type: Literal["all_missing"]
    missing_rate: float


_ColumnModel = _NumericModel | _DiscreteModel | _AllMissingModel


class BaselineGenerator(BaseGenerator):
    """Minimal baseline generator for Phase 2.

    - Numeric columns: sample from Normal(mean, std).
    - Discrete columns (categorical/label/datetime/text): sample from observed distribution.
    - Missing values: reproduce per-column missing rate.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        guard: GuardEngine | None = None,
        scenario: ScenarioEngine | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self._column_order: list[str] = []
        self._models: dict[str, _ColumnModel] = {}
        self.guard = guard
        self.scenario = scenario

    def fit(self, real: pd.DataFrame) -> None:
        if real.empty:
            raise ValueError("Real dataset must contain at least one row.")
        if len(real.columns) == 0:
            raise ValueError("Real dataset must contain at least one column.")

        detector = TypeDetector()
        self._column_order = [str(column) for column in real.columns]
        self._models = {}

        for column in self._column_order:
            series = real[column]
            self._models[column] = self._build_column_model(series, column, detector)

    def sample(self, n_samples: int) -> pd.DataFrame:
        if not self._models or not self._column_order:
            raise RuntimeError("Generator is not fitted. Call fit() before sample().")

        rng = np.random.default_rng(self.seed)
        if n_samples <= 0:
            return self._sample_batch(rng, 0)

        if self.guard and self.guard.fill_to_target and self.guard.policy.value == "exclude":
            return self._sample_with_fill(rng, n_samples)

        df = self._sample_batch(rng, n_samples)

        if self.scenario:
            df, _ = self.scenario.apply(df)

        if self.guard:
            df, _ = self.guard.apply(df)

        return df

    def _sample_with_fill(self, rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
        """Sample with guard EXCLUDE mode, resampling until n_samples rows pass constraints."""
        guard = self.guard  # guard is guaranteed non-None by the caller
        if guard is None:
            return self._sample_batch(rng, 0)

        max_batch_size = max(n_samples * 128, 1024)
        max_empty_batches_at_cap = 5
        batch_size = max(n_samples * 2, n_samples)
        accumulated: list[pd.DataFrame] = []
        accumulated_count = 0
        consecutive_empty_batches = 0
        empty_result: pd.DataFrame | None = None

        while accumulated_count < n_samples:
            batch_df = self._sample_batch(rng, batch_size)

            if self.scenario:
                batch_df, _ = self.scenario.apply(batch_df)

            batch_df, _ = guard.apply(batch_df)
            empty_result = batch_df.iloc[0:0].copy()

            if batch_df.empty:
                consecutive_empty_batches += 1
                if (
                    batch_size >= max_batch_size
                    and consecutive_empty_batches >= max_empty_batches_at_cap
                ):
                    if accumulated_count > 0:
                        raise RuntimeError(
                            "Unable to reach target sample size because the guard stopped "
                            "admitting rows during repeated refill attempts."
                        )
                    return empty_result
                batch_size = min(batch_size * 2, max_batch_size)
                continue

            accumulated.append(batch_df)
            accumulated_count += len(batch_df)
            consecutive_empty_batches = 0

            needed = n_samples - accumulated_count
            if needed <= 0:
                break

            keep_rate = len(batch_df) / batch_size
            estimated_batch_size = math.ceil((needed / keep_rate) * 1.1)
            batch_size = min(max(needed, estimated_batch_size), max_batch_size)

        if not accumulated:
            return empty_result if empty_result is not None else self._sample_batch(rng, 0)

        combined: pd.DataFrame = pd.concat(accumulated, ignore_index=True)
        result: pd.DataFrame = combined.iloc[:n_samples].reset_index(drop=True)
        return result

    def _sample_batch(self, rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
        data: dict[str, object] = {}
        for column in self._column_order:
            model = self._models[column]
            data[column] = self._sample_column(model, rng, n_samples)
        return pd.DataFrame(data, columns=self._column_order)

    def _build_column_model(
        self, series: pd.Series, column: str, detector: TypeDetector
    ) -> _ColumnModel:
        missing_rate = float(series.isna().mean())
        non_null = series.dropna()
        if non_null.empty:
            return _AllMissingModel(model_type="all_missing", missing_rate=1.0)

        detected = detector.detect(series, column)

        if detected == ColumnType.NUMERIC:
            if self._looks_like_discrete_numeric(non_null) or self._looks_like_id_column(column):
                detected = ColumnType.CATEGORICAL

        if detected == ColumnType.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return _AllMissingModel(model_type="all_missing", missing_rate=1.0)

            mean = float(numeric.mean())
            std = float(numeric.std(ddof=0))
            if not np.isfinite(std):
                std = 0.0

            constant_value = float(numeric.iloc[0]) if std == 0.0 else None
            return _NumericModel(
                model_type="numeric",
                missing_rate=missing_rate,
                mean=mean,
                std=std,
                constant_value=constant_value,
            )

        counts = non_null.value_counts(dropna=True)
        choices = list(counts.index)
        probabilities = (counts / counts.sum()).astype(float).tolist()
        return _DiscreteModel(
            model_type="discrete",
            missing_rate=missing_rate,
            choices=choices,
            probabilities=probabilities,
        )

    @staticmethod
    def _looks_like_discrete_numeric(series: pd.Series) -> bool:
        total = len(series)
        if total == 0:
            return False
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / total
        return unique_count <= 20 and unique_ratio <= 0.5

    @staticmethod
    def _looks_like_id_column(column: str) -> bool:
        lowered = column.lower()
        return lowered == "id" or lowered.endswith("_id") or lowered.endswith("id")

    @staticmethod
    def _apply_missing_mask(
        rng: np.random.Generator, values: NDArray[np.generic], missing_rate: float
    ) -> NDArray[np.generic]:
        if missing_rate <= 0.0:
            return values

        mask = rng.random(len(values)) < missing_rate
        values = values.copy()
        values[mask] = np.nan
        return values

    def _sample_column(
        self, model: _ColumnModel, rng: np.random.Generator, n_samples: int
    ) -> NDArray[np.generic]:
        if model.model_type == "all_missing":
            return np.full(n_samples, np.nan, dtype=object)

        if model.model_type == "numeric":
            if model.constant_value is not None:
                values = np.full(n_samples, model.constant_value, dtype=float)
            else:
                std = model.std if model.std > 0.0 else 1e-6
                values = rng.normal(model.mean, std, size=n_samples).astype(float)
            return self._apply_missing_mask(rng, values, model.missing_rate)

        choices = np.asarray(model.choices, dtype=object)
        probabilities = np.asarray(model.probabilities, dtype=float)
        values = rng.choice(choices, size=n_samples, p=probabilities).astype(object)
        return self._apply_missing_mask(rng, values, model.missing_rate)
