from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sfdao.generator.ctgan import CTGANGenerator


@pytest.fixture
def mock_sdv():
    with (
        patch("sfdao.generator.ctgan.CTGANSynthesizer") as mock_synth,
        patch("sfdao.generator.ctgan.SingleTableMetadata") as mock_meta,
    ):
        yield mock_synth, mock_meta


class TestCTGANGenerator:
    def test_init_raises_import_error_if_sdv_missing(self):
        with patch("sfdao.generator.ctgan.CTGANSynthesizer", None):
            with pytest.raises(ImportError, match="requires 'sdv' package"):
                CTGANGenerator()

    def test_init_success(self, mock_sdv):
        gen = CTGANGenerator(seed=42)
        assert gen.seed == 42
        assert gen.guard is None
        assert gen.scenario is None

    def test_fit_calls_sdv_fit(self, mock_sdv):
        mock_synth_cls, mock_meta_cls = mock_sdv
        gen = CTGANGenerator()
        real_data = pd.DataFrame({"col": [1, 2, 3]})

        gen.fit(real_data)

        # Verified metadata detection called
        mock_meta_cls.return_value.detect_from_dataframe.assert_called_with(real_data)
        # Verified fit called
        mock_synth_cls.return_value.fit.assert_called_with(real_data)

    def test_fit_raises_on_empty(self, mock_sdv):
        gen = CTGANGenerator()
        with pytest.raises(ValueError, match="must contain at least one row"):
            gen.fit(pd.DataFrame())

    def test_sample_calls_sdv_sample(self, mock_sdv):
        mock_synth_cls, _ = mock_sdv
        gen = CTGANGenerator()
        gen.fit(pd.DataFrame({"col": [1, 2, 3]}))  # needed to init synthesizer

        mock_instance = mock_synth_cls.return_value
        expected_df = pd.DataFrame({"col": [1, 1]})
        mock_instance.sample.return_value = expected_df

        sampled = gen.sample(2)

        mock_instance.sample.assert_called_with(num_rows=2)
        assert sampled.equals(expected_df)

    def test_sample_raises_if_not_fitted(self, mock_sdv):
        gen = CTGANGenerator()
        with pytest.raises(RuntimeError, match="not fitted"):
            gen.sample(10)

    def test_sample_applies_scenario_and_guard(self, mock_sdv):
        mock_synth_cls, _ = mock_sdv

        # Mock Scenario
        mock_scenario = MagicMock()
        mock_scenario.apply.side_effect = lambda df: (df.assign(scen=1), {})

        # Mock Guard
        mock_guard = MagicMock()
        mock_guard.apply.side_effect = lambda df: (df.assign(guard=1), {})

        gen = CTGANGenerator(scenario=mock_scenario, guard=mock_guard)
        gen.fit(pd.DataFrame({"col": [1]}))

        mock_instance = mock_synth_cls.return_value
        mock_instance.sample.return_value = pd.DataFrame({"col": [10]})

        result = gen.sample(5)

        assert "scen" in result.columns
        assert "guard" in result.columns
        mock_scenario.apply.assert_called_once()
        mock_guard.apply.assert_called_once()
