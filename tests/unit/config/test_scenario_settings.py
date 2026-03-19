"""Tests for simplified ScenarioSettings structure (Issue #39)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sfdao.config.loader import load_phase2_config
from sfdao.config.models import ScenarioSettings


def test_scenario_settings_direct_fields():
    """ScenarioSettings should accept name and transformations directly."""
    settings = ScenarioSettings(
        enabled=True,
        name="Test Scenario",
        transformations=[{"column": "amount", "type": "scale", "params": {"factor": 2.0}}],
    )
    assert settings.enabled is True
    assert settings.name == "Test Scenario"
    assert len(settings.transformations) == 1
    assert settings.transformations[0].column == "amount"
    assert settings.transformations[0].type == "scale"
    assert settings.transformations[0].params["factor"] == 2.0


def test_scenario_settings_enabled_false_no_name():
    """ScenarioSettings with enabled=False and no name/transformations is valid."""
    settings = ScenarioSettings(enabled=False)
    assert settings.enabled is False
    assert settings.name is None
    assert settings.transformations is None


def test_scenario_settings_rejects_unknown_fields():
    """ScenarioSettings should reject unknown fields (extra='forbid')."""
    with pytest.raises(ValidationError):
        ScenarioSettings(enabled=True, unknown_field="bad")  # type: ignore[call-arg]


def test_scenario_settings_rejects_params_key():
    """Old-style 'params' key should be rejected since it was replaced by direct fields."""
    with pytest.raises(ValidationError):
        ScenarioSettings(  # type: ignore[call-arg]
            enabled=True,
            params={
                "name": "Old Style",
                "transformations": [],
            },
        )


def test_load_phase2_config_with_simplified_scenario(tmp_path: pytest.TempPathFactory) -> None:
    """YAML config should support simplified scenario structure."""
    config_path = tmp_path / "phase2.yaml"  # type: ignore
    config_path.write_text(
        "\n".join(
            [
                "version: 2",
                "generator:",
                "  type: baseline",
                "  n_samples: 100",
                "scenario:",
                "  enabled: true",
                "  name: My Scenario",
                "  transformations:",
                "    - column: amount",
                "      type: shift",
                "      params:",
                "        value: 500.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_phase2_config(config_path)
    assert config.scenario is not None
    assert config.scenario.enabled is True
    assert config.scenario.name == "My Scenario"
    assert config.scenario.transformations is not None
    assert len(config.scenario.transformations) == 1
    assert config.scenario.transformations[0].column == "amount"
    assert config.scenario.transformations[0].type == "shift"
    assert config.scenario.transformations[0].params["value"] == 500.0
