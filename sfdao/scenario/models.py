from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ScenarioConfig", "TransformationConfig"]

TransformationType = Literal["scale", "shift", "clip", "outlier", "replace"]


class TransformationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    column: str = Field(..., min_length=1, description="Target column name")
    type: TransformationType = Field(
        ..., description="Transformation type (scale, shift, clip, outlier, replace)"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the transformation"
    )
    condition: dict[str, Any] | None = Field(None, description="Optional condition")


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(..., min_length=1, description="Scenario name")
    transformations: list[TransformationConfig] = Field(..., description="List of transformations")
