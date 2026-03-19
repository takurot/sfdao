"""Integration tests: GeneratorSettings.params flows through to generator instances.

Covers the fix for Issue #37 where params specified in YAML/config were silently
ignored when build_generator instantiated BaselineGenerator or CTGANGenerator.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pandas as pd

from sfdao.config.loader import load_phase2_config
from sfdao.config.models import Phase2Config
from sfdao.generator.baseline import BaselineGenerator
from sfdao.generator.ctgan import CTGANGenerator
from sfdao.generator.factory import build_generator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_phase2_config(tmp_path: Path, yaml_text: str) -> Phase2Config:
    config_path = tmp_path / "phase2.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")
    return load_phase2_config(config_path)


# ---------------------------------------------------------------------------
# Tests: params reach generator via build_generator
# ---------------------------------------------------------------------------


class TestGeneratorParamsFromConfig:
    """Verify that generator.params defined in config reach the generator."""

    def test_baseline_params_stored_from_config(self, tmp_path: Path):
        cfg = _load_phase2_config(
            tmp_path,
            dedent("""\
            version: 2
            seed: 1
            generator:
              type: baseline
              n_samples: 10
              params:
                custom_key: hello
                numeric_param: 99
            """),
        )
        gen = build_generator(cfg.generator, seed=cfg.seed)
        assert isinstance(gen, BaselineGenerator)
        assert gen.params == {"custom_key": "hello", "numeric_param": 99}

    def test_ctgan_params_stored_from_config(self, tmp_path: Path):
        with patch("sfdao.generator.ctgan.CTGANSynthesizer"):
            cfg = _load_phase2_config(
                tmp_path,
                dedent("""\
                version: 2
                seed: 7
                generator:
                  type: ctgan
                  n_samples: 50
                  params:
                    epochs: 100
                    batch_size: 256
                """),
            )
            gen = build_generator(cfg.generator, seed=cfg.seed)
            assert isinstance(gen, CTGANGenerator)
            assert gen.params == {"epochs": 100, "batch_size": 256}
            assert gen.seed == 7

    def test_ctgan_params_passed_to_synthesizer_on_fit(self, tmp_path: Path):
        """Params from config must reach CTGANSynthesizer.__init__ at fit time."""
        with (
            patch("sfdao.generator.ctgan.CTGANSynthesizer") as mock_synth,
            patch("sfdao.generator.ctgan.SingleTableMetadata"),
        ):
            cfg = _load_phase2_config(
                tmp_path,
                dedent("""\
                version: 2
                generator:
                  type: ctgan
                  n_samples: 5
                  params:
                    epochs: 30
                    batch_size: 64
                """),
            )
            gen = build_generator(cfg.generator, seed=None)
            real_df = pd.DataFrame({"x": [1, 2, 3]})
            gen.fit(real_df)

            call_kwargs = mock_synth.call_args.kwargs
            assert call_kwargs.get("epochs") == 30
            assert call_kwargs.get("batch_size") == 64

    def test_empty_params_is_ok(self, tmp_path: Path):
        cfg = _load_phase2_config(
            tmp_path,
            dedent("""\
            version: 2
            generator:
              type: baseline
              n_samples: 5
            """),
        )
        gen = build_generator(cfg.generator, seed=None)
        assert gen.params == {}

    def test_baseline_generates_samples_with_params_present(self, tmp_path: Path):
        """Smoke test: BaselineGenerator still produces correct output when params set."""
        cfg = _load_phase2_config(
            tmp_path,
            dedent("""\
            version: 2
            seed: 42
            generator:
              type: baseline
              n_samples: 20
              params:
                ignored_by_baseline: true
            """),
        )
        gen = build_generator(cfg.generator, seed=cfg.seed)
        real_df = pd.DataFrame({"amount": [10.0, 20.0, 30.0]})
        gen.fit(real_df)
        synth = gen.sample(cfg.generator.n_samples)
        assert len(synth) == 20
        assert list(synth.columns) == ["amount"]
