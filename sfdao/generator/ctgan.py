from __future__ import annotations

import pandas as pd

from sfdao.generator.base import BaseGenerator
from sfdao.guard.engine import GuardEngine
from sfdao.scenario.engine import ScenarioEngine

try:
    from sdv.metadata import SingleTableMetadata  # type: ignore
    from sdv.single_table import CTGANSynthesizer  # type: ignore
except ImportError:
    SingleTableMetadata = None
    CTGANSynthesizer = None


class CTGANGenerator(BaseGenerator):
    """Deep learning based generator using CTGAN (via SDV)."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        guard: GuardEngine | None = None,
        scenario: ScenarioEngine | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if CTGANSynthesizer is None:
            raise ImportError(
                "CTGANGenerator requires 'sdv' package. "
                "Please install it with `pip install sfdao[deep]`."
            )

        self.guard = guard
        self.scenario = scenario
        self._synthesizer: CTGANSynthesizer | None = None

    def fit(self, real: pd.DataFrame) -> None:
        if real.empty:
            raise ValueError("Real dataset must contain at least one row.")

        # Auto-detect metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real)

        # Initialize synthesizer with fixed epochs for now (can be configurable later)
        # Using a small number of epochs by default if not specified via some config mechanism,
        # but SDV defaults (300) might be too slow for quick tests.
        # Ideally, we should pass parameters from settings.
        # For now, stick to defaults but allow seed.

        # Note: SDV 1.x CTGANSynthesizer does not accept random_state/seed in __init__ easily
        # in the same way, but it uses torch/numpy seeds.
        # We can set verbose=True to see progress.
        self._synthesizer = CTGANSynthesizer(
            metadata,
            enforce_rounding=True,
            enforce_min_max_values=True,
            verbose=False,
        )
        # fit() in SDV doesn't take seed directly usually.
        self._synthesizer.fit(real)

    def sample(self, n_samples: int) -> pd.DataFrame:
        if self._synthesizer is None:
            raise RuntimeError("Generator is not fitted. Call fit() before sample().")

        # Set seed context if possible, or reliance on global seed if SDV uses it?
        # SDV's sample() method doesn't take a seed argument in recent versions usually?
        # Actually it does not.
        # But we can try to set global seeds if needed,
        # but let's just straightforwardly call sample.
        df = pd.DataFrame(self._synthesizer.sample(num_rows=n_samples))

        if self.scenario:
            df, _ = self.scenario.apply(df)

        if self.guard:
            df, _ = self.guard.apply(df)

        return df
