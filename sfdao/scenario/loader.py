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

    name = settings.name
    transformations = settings.transformations
    if name is None or transformations is None:
        raise ValueError("Enabled scenario settings must define name and transformations")
    config = ScenarioConfig(
        name=name,
        transformations=transformations,
    )
    return ScenarioEngine(config, seed=seed)
