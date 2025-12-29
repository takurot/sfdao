from __future__ import annotations

from sfdao.config.models import GeneratorSettings
from sfdao.generator.base import BaseGenerator
from sfdao.generator.baseline import BaselineGenerator
from sfdao.guard.engine import GuardEngine

__all__ = ["build_generator"]


def build_generator(
    settings: GeneratorSettings, *, seed: int | None, guard: GuardEngine | None = None
) -> BaseGenerator:
    generator_type = settings.type.strip().lower()
    if generator_type == "baseline":
        return BaselineGenerator(seed=seed, guard=guard)

    raise ValueError(f"Unsupported generator type: {settings.type}")
