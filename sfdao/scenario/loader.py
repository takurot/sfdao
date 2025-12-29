from __future__ import annotations

from sfdao.config.models import ScenarioSettings
from sfdao.scenario.engine import ScenarioEngine
from sfdao.scenario.models import ScenarioConfig

__all__ = ["load_scenario_engine"]


def load_scenario_engine(
    settings: ScenarioSettings | None, seed: int | None = None
) -> ScenarioEngine | None:
    if not settings or not settings.enabled:
        return None

    # settings.params must match ScenarioConfig structure
    config = ScenarioConfig.model_validate(settings.params)
    return ScenarioEngine(config, seed=seed)
