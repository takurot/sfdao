from __future__ import annotations

from sfdao.config.models import GeneratorSettings
from sfdao.generator.base import BaseGenerator
from sfdao.generator.baseline import BaselineGenerator

__all__ = ["build_generator"]


def build_generator(settings: GeneratorSettings, *, seed: int | None) -> BaseGenerator:
    generator_type = settings.type.strip().lower()
    if generator_type == "baseline":
        return BaselineGenerator(seed=seed)

    raise ValueError(f"Unsupported generator type: {settings.type}")
