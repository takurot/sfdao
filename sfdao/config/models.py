from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AuditSettings",
    "GeneratorSettings",
    "GuardSettings",
    "Phase2Config",
    "ScenarioSettings",
]


class GeneratorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: str = Field(..., min_length=1, description="Generator type identifier (e.g., baseline).")
    n_samples: int = Field(
        ...,
        ge=1,
        description="Number of synthetic rows to generate.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Generator-specific configuration parameters.",
    )


class GuardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = Field(True, description="Whether constraint/logic guard is enabled.")
    mode: Literal["detect", "exclude", "clip", "correct"] = Field(
        "detect", description="How to handle violations (detect only, drop rows, clip values)."
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Guard-specific parameters.")


class ScenarioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = Field(False, description="Whether scenario injection is enabled.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Scenario-specific parameters."
    )


class PrivacySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Sample size for expensive privacy metric calculations.",
    )


class AuditSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    weights: dict[str, float] | None = Field(
        default=None,
        description="Optional component weights override (e.g., quality/utility/privacy).",
    )
    privacy: PrivacySettings | None = Field(
        default=None,
        description="Privacy evaluation settings.",
    )


class Phase2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: int = Field(2, ge=1, description="Config schema version.")
    seed: int | None = Field(
        default=None, ge=0, description="Global random seed for reproducibility."
    )
    generator: GeneratorSettings
    guard: GuardSettings | None = None
    scenario: ScenarioSettings | None = None
    audit: AuditSettings | None = None
