from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from sfdao.scenario.models import ScenarioConfig
from sfdao.scenario.transformations import TransformationRegistry

__all__ = ["ScenarioEngine"]

logger = logging.getLogger(__name__)


class ScenarioEngine:
    def __init__(self, config: ScenarioConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        # Work on a copy
        df = df.copy()

        applied_log = []

        for tx in self.config.transformations:
            if tx.column not in df.columns:
                logger.warning(
                    "Column '%s' not found in DataFrame, skipping transformation '%s'",
                    tx.column,
                    tx.type,
                )
                continue

            func = TransformationRegistry.get(tx.type)
            series = df[tx.column]

            # Apply transformation
            # We pass our RNG state (or separate RNG per step if we want deterministic sequence)
            # Sharing one RNG is better for sequence.
            new_series = func(series, tx.params, self.rng)
            df[tx.column] = new_series

            applied_log.append({"column": tx.column, "type": tx.type, "params": tx.params})

        metadata = {"scenario": {"name": self.config.name, "applied": applied_log}}

        return df, metadata
