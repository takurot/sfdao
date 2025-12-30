from unittest.mock import patch

import pytest
from sfdao.config.models import GeneratorSettings
from sfdao.generator.baseline import BaselineGenerator
from sfdao.generator.ctgan import CTGANGenerator
from sfdao.generator.factory import build_generator


class TestGeneratorFactory:
    def test_build_baseline(self):
        settings = GeneratorSettings(type="baseline", n_samples=100)
        gen = build_generator(settings, seed=42)
        assert isinstance(gen, BaselineGenerator)
        assert gen.seed == 42

    def test_build_ctgan(self):
        # We need to ensure CTGANGenerator can be instantiated (mock sdv presence if needed)
        # But import check is in __init__. So instantiation raises ImportError if sdv missing.
        # We should check that build_generator RETURNS a CTGANGenerator instance (or tries to).
        with patch("sfdao.generator.ctgan.CTGANSynthesizer"):
            settings = GeneratorSettings(type="ctgan", n_samples=100)
            gen = build_generator(settings, seed=123)
            assert isinstance(gen, CTGANGenerator)
            assert gen.seed == 123

    def test_build_unknown_raises(self):
        settings = GeneratorSettings(type="unknown", n_samples=100)
        with pytest.raises(ValueError, match="Unsupported generator type"):
            build_generator(settings, seed=42)
