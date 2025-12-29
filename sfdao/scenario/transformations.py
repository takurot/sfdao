from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

__all__ = ["TransformationRegistry"]


TransformationFunc = Callable[[pd.Series, dict[str, Any], np.random.Generator | None], pd.Series]


class TransformationRegistry:
    _registry: dict[str, TransformationFunc] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[TransformationFunc], TransformationFunc]:
        def decorator(func: TransformationFunc) -> TransformationFunc:
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> TransformationFunc:
        if name not in cls._registry:
            raise ValueError(f"Unknown transformation: {name}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())


@TransformationRegistry.register("scale")
def scale(
    series: pd.Series,
    params: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.Series:
    factor = params.get("factor", 1.0)
    # Ensure numeric
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric * factor
    return result  # type: ignore[no-any-return]


@TransformationRegistry.register("shift")
def shift(
    series: pd.Series,
    params: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.Series:
    value = params.get("value", 0.0)
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric + value
    return result  # type: ignore[no-any-return]


@TransformationRegistry.register("clip")
def clip(
    series: pd.Series,
    params: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.Series:
    min_val = params.get("min")
    max_val = params.get("max")
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.clip(lower=min_val, upper=max_val)


@TransformationRegistry.register("replace")
def replace(
    series: pd.Series,
    params: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.Series:
    old = params.get("old")
    new = params.get("new")
    if old is None:
        return series
    return series.replace(old, new)


@TransformationRegistry.register("outlier")
def outlier(
    series: pd.Series,
    params: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> pd.Series:
    n = params.get("n", 0)
    value = params.get("value")

    if value is None:
        raise ValueError("'value' parameter is required for outlier transformation")

    if n <= 0:
        return series

    if rng is None:
        rng = np.random.default_rng()

    indices = rng.choice(series.index, size=min(n, len(series)), replace=False)

    # We need to copy to avoid modifying original if it's passed by ref
    result = series.copy()
    result.loc[indices] = value

    return result
