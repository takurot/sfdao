import pandas as pd
import pytest

from sfdao.generator.ctgan import CTGANGenerator

try:
    import sdv  # noqa: F401

    HAS_SDV = True
except ImportError:
    HAS_SDV = False


@pytest.mark.skipif(not HAS_SDV, reason="sdv not installed")
@pytest.mark.e2e
def test_ctgan_smoke():
    # Very small dataset for smoke test
    real = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0] * 10, "B": ["x", "y", "x", "y", "z"] * 10})

    gen = CTGANGenerator(seed=42)

    # Check if fit runs without error
    # This might be slow if CTGAN actually trains default epochs
    # We rely on defaults being reasonable or just slow but working.
    # In real CI, might need config to reduce epochs.
    # But for now, we just run it.
    gen.fit(real)

    sampled = gen.sample(10)

    assert len(sampled) == 10
    assert list(sampled.columns) == list(real.columns)
    assert set(sampled["B"].unique()).issubset({"x", "y", "z"})
