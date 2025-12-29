import pytest
from typer.testing import CliRunner

from sfdao.cli.main import app

runner = CliRunner()


def test_generate_validate_only_with_valid_config(tmp_path: pytest.TempPathFactory) -> None:
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

    result = runner.invoke(app, ["generate", "--config", str(config_path), "--validate-only"])
    assert result.exit_code == 0
    assert "config" in result.output.lower()


def test_run_validate_only_with_valid_config(tmp_path: pytest.TempPathFactory) -> None:
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

    result = runner.invoke(app, ["run", "--config", str(config_path), "--validate-only"])
    assert result.exit_code == 0
    assert "config" in result.output.lower()


def test_generate_rejects_invalid_config(tmp_path: pytest.TempPathFactory) -> None:
    config_path = tmp_path / "phase2.yaml"  # type: ignore
    config_path.write_text("version: 2\nunknown_field: 1\n", encoding="utf-8")

    result = runner.invoke(app, ["generate", "--config", str(config_path), "--validate-only"])
    assert result.exit_code != 0
    assert "unknown" in result.output.lower() or "extra" in result.output.lower()
