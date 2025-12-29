from __future__ import annotations

from pathlib import Path

import pandas as pd

from sfdao.generator.baseline import BaselineGenerator
from sfdao.ingestion.loader import CSVLoader


def test_baseline_generator_is_reproducible_with_seed() -> None:
    real = pd.DataFrame(
        {
            "amount": [1.0, 2.5, 3.5, 4.0, 5.0],
            "category": ["A", "B", "A", "A", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-04T00:00:00Z",
                    "2024-01-05T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    generator_a = BaselineGenerator(seed=123)
    generator_a.fit(real)
    sample_a = generator_a.sample(50)

    generator_b = BaselineGenerator(seed=123)
    generator_b.fit(real)
    sample_b = generator_b.sample(50)

    pd.testing.assert_frame_equal(sample_a, sample_b)


def test_baseline_generator_preserves_schema_and_handles_missing_and_constant() -> None:
    real = pd.DataFrame(
        {
            "amount": [1.0, 2.0, None, 4.0, 5.0],
            "category": ["a", "b", "a", None, "b"],
            "constant": [7, 7, 7, 7, 7],
            "all_missing": [None, None, None, None, None],
        }
    )

    generator = BaselineGenerator(seed=42)
    generator.fit(real)
    synthetic = generator.sample(100)

    assert list(synthetic.columns) == list(real.columns)
    assert len(synthetic) == 100
    assert set(synthetic["category"].dropna().unique()).issubset({"a", "b"})
    assert set(synthetic["constant"].unique()) == {7}
    assert synthetic["all_missing"].isna().all()
    assert pd.api.types.is_numeric_dtype(synthetic["amount"])


def test_baseline_generator_output_roundtrips_via_csvloader(tmp_path: Path) -> None:
    real = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                None,
                "2024-01-04T00:00:00Z",
            ],
            "label": ["x", "y", "x", "y"],
        }
    )

    generator = BaselineGenerator(seed=7)
    generator.fit(real)
    synthetic = generator.sample(25)

    output_path = tmp_path / "synthetic.csv"
    synthetic.to_csv(output_path, index=False)

    reloaded = CSVLoader().load(output_path)
    assert list(reloaded.columns) == list(real.columns)
    assert len(reloaded) == 25
