from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from sfdao.config.models import Phase2Config

__all__ = ["load_phase2_config"]


def load_phase2_config(path: str | Path) -> Phase2Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    payload = config_path.read_text(encoding="utf-8")
    data = _parse_config_payload(config_path, payload)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top-level: {config_path}")

    return Phase2Config.model_validate(data)


def _parse_config_payload(config_path: Path, payload: str) -> Any:
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(payload)
    if suffix == ".json":
        return json.loads(payload)

    raise ValueError("Unsupported config format. Use .yaml/.yml or .json: " f"{config_path.name}")
