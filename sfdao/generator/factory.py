from __future__ import annotations

from sfdao.config.models import GeneratorSettings
from sfdao.generator.base import BaseGenerator
from sfdao.generator.baseline import BaselineGenerator
from sfdao.generator.ctgan import CTGANGenerator
from sfdao.guard.engine import GuardEngine
from sfdao.scenario.engine import ScenarioEngine

__all__ = ["build_generator"]


def build_generator(
    settings: GeneratorSettings,
    *,
    seed: int | None,
    guard: GuardEngine | None = None,
    scenario: ScenarioEngine | None = None,
) -> BaseGenerator:
    generator_type = settings.type.strip().lower()
    if generator_type == "baseline":
        return BaselineGenerator(seed=seed, guard=guard, scenario=scenario)

    if generator_type == "ctgan":
        return CTGANGenerator(seed=seed, guard=guard, scenario=scenario)

    raise ValueError(f"Unsupported generator type: {settings.type}")
