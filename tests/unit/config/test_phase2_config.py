import pytest
from pydantic import ValidationError

from sfdao.config.loader import load_phase2_config


def test_load_phase2_config_from_yaml(tmp_path: pytest.TempPathFactory) -> None:
    config_path = tmp_path / "phase2.yaml"  # type: ignore
    config_path.write_text(
        "\n".join(
            [
                "version: 2",
                "seed: 42",
                "generator:",
                "  type: baseline",
                "  n_samples: 1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_phase2_config(config_path)
    assert config.version == 2
    assert config.seed == 42
    assert config.generator.type == "baseline"
    assert config.generator.n_samples == 1000


def test_load_phase2_config_rejects_unknown_fields(tmp_path: pytest.TempPathFactory) -> None:
    config_path = tmp_path / "phase2.yaml"  # type: ignore
    config_path.write_text("version: 2\nunknown_field: 1\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_phase2_config(config_path)
