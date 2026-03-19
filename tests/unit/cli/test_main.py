"""Unit tests for CLI main module.

TDD - Red phase: Writing tests before implementation.
"""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import sfdao.cli.main as cli_main

app = cli_main.app

runner = CliRunner()


class TestCliBasic:
    """Basic CLI functionality tests."""

    @pytest.mark.skip(reason="Typer 0.15 has a known issue with Optional[Path] in --help display")
    def test_cli_help_shows_available_commands(self) -> None:
        """Test that CLI --help shows available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "audit" in result.output

    def test_cli_version_option(self) -> None:
        """Test that CLI has version option."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestAuditCommand:
    """Tests for the 'audit' subcommand."""

    def test_audit_requires_real_and_synthetic_args(self) -> None:
        """Test that audit command requires --real and --synthetic arguments."""
        result = runner.invoke(app, ["audit"])
        assert result.exit_code != 0
        assert "--real" in result.output or "Missing" in result.output

    def test_audit_with_valid_args(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test audit command with valid file arguments."""
        # Create test CSV files
        real_csv = tmp_path / "real.csv"  # type: ignore
        synthetic_csv = tmp_path / "synthetic.csv"  # type: ignore
        output_file = tmp_path / "report.txt"  # type: ignore

        # Simple test data
        real_csv.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n")
        synthetic_csv.write_text("col1,col2\n1.1,2.1\n3.1,4.1\n5.1,6.1\n7.1,8.1\n9.1,10.1\n")

        result = runner.invoke(
            app,
            [
                "audit",
                "--real",
                str(real_csv),
                "--synthetic",
                str(synthetic_csv),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_audit_fails_with_missing_real_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test audit command fails when real file doesn't exist."""
        synthetic_csv = tmp_path / "synthetic.csv"  # type: ignore
        synthetic_csv.write_text("col1,col2\n1.0,2.0\n")

        result = runner.invoke(
            app,
            [
                "audit",
                "--real",
                "/nonexistent/real.csv",
                "--synthetic",
                str(synthetic_csv),
            ],
        )

        assert result.exit_code != 0
        # Typer outputs "does not exist" with line break
        assert "does not" in result.output

    def test_audit_fails_with_missing_synthetic_file(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test audit command fails when synthetic file doesn't exist."""
        real_csv = tmp_path / "real.csv"  # type: ignore
        real_csv.write_text("col1,col2\n1.0,2.0\n")

        result = runner.invoke(
            app,
            [
                "audit",
                "--real",
                str(real_csv),
                "--synthetic",
                "/nonexistent/synthetic.csv",
            ],
        )

        assert result.exit_code != 0
        # Typer outputs "does not exist" with line break
        assert "does not" in result.output


class TestAuditOutputFormats:
    """Tests for audit command output format handling."""

    def test_audit_default_output_format_is_txt(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that default output format is plain text."""
        real_csv = tmp_path / "real.csv"  # type: ignore
        synthetic_csv = tmp_path / "synthetic.csv"  # type: ignore
        output_file = tmp_path / "report.txt"  # type: ignore

        real_csv.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n")
        synthetic_csv.write_text("col1,col2\n1.1,2.1\n3.1,4.1\n5.1,6.1\n7.1,8.1\n9.1,10.1\n")

        result = runner.invoke(
            app,
            [
                "audit",
                "--real",
                str(real_csv),
                "--synthetic",
                str(synthetic_csv),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        content = output_file.read_text()
        assert "Overall Score" in content
        assert "privacy_risk" in content
        assert "financial_facts" in content


class TestAuditVerbosity:
    """Tests for audit command verbosity options."""

    def test_audit_quiet_mode(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that quiet mode suppresses console output."""
        real_csv = tmp_path / "real.csv"  # type: ignore
        synthetic_csv = tmp_path / "synthetic.csv"  # type: ignore
        output_file = tmp_path / "report.txt"  # type: ignore

        real_csv.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n")
        synthetic_csv.write_text("col1,col2\n1.1,2.1\n3.1,4.1\n5.1,6.1\n7.1,8.1\n9.1,10.1\n")

        result = runner.invoke(
            app,
            [
                "audit",
                "--real",
                str(real_csv),
                "--synthetic",
                str(synthetic_csv),
                "--output",
                str(output_file),
                "--quiet",
            ],
        )

        assert result.exit_code == 0
        # In quiet mode, minimal output
        assert len(result.output.strip()) < 100 or result.output.strip() == ""


class TestRunCommand:
    """Tests for the 'run' subcommand, specifically ML utility flags."""

    def test_run_accepts_ml_utility_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that run command accepts --ml-utility flag."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "generator:\n  type: gaussian\n  n_samples: 5\n",
            encoding="utf-8",
        )
        real_csv = tmp_path / "real.csv"
        real_csv.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")

        calls: list[dict] = []

        def fake_run_audit(**kwargs):  # noqa: ANN001
            calls.append(kwargs)

        monkeypatch.setattr(cli_main, "run_audit", fake_run_audit)

        import sfdao.cli.main as m

        monkeypatch.setattr(m, "build_generator", lambda *a, **kw: _FakeGenerator(real_csv))

        result = runner.invoke(
            app,
            [
                "run",
                "--real",
                str(real_csv),
                "--config",
                str(config_file),
                "--out-dir",
                str(tmp_path / "out"),
                "--ml-utility",
                "--ml-target",
                "col1",
                "--quiet",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0]["ml_utility"] is True
        assert calls[0]["ml_target"] == "col1"

    def test_run_ml_utility_requires_ml_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that run command fails when --ml-utility is set without --ml-target."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "generator:\n  type: gaussian\n  n_samples: 5\n",
            encoding="utf-8",
        )
        real_csv = tmp_path / "real.csv"
        real_csv.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "run",
                "--real",
                str(real_csv),
                "--config",
                str(config_file),
                "--out-dir",
                str(tmp_path / "out"),
                "--ml-utility",
                "--quiet",
            ],
        )
        assert result.exit_code != 0
        assert "ml-target" in result.output.lower() or "required" in result.output.lower()

    def test_run_without_ml_utility_passes_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that run command passes ml_utility=False when flag is absent."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "generator:\n  type: gaussian\n  n_samples: 5\n",
            encoding="utf-8",
        )
        real_csv = tmp_path / "real.csv"
        real_csv.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")

        calls: list[dict] = []

        def fake_run_audit(**kwargs):  # noqa: ANN001
            calls.append(kwargs)

        monkeypatch.setattr(cli_main, "run_audit", fake_run_audit)

        import sfdao.cli.main as m

        monkeypatch.setattr(m, "build_generator", lambda *a, **kw: _FakeGenerator(real_csv))

        result = runner.invoke(
            app,
            [
                "run",
                "--real",
                str(real_csv),
                "--config",
                str(config_file),
                "--out-dir",
                str(tmp_path / "out"),
                "--quiet",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0]["ml_utility"] is False
        assert calls[0]["ml_target"] is None


class _FakeGenerator:
    """Minimal fake generator for tests."""

    def __init__(self, real_csv: Path) -> None:
        import pandas as pd

        self._df = pd.read_csv(real_csv)

    def fit(self, df) -> None:  # noqa: ANN001
        pass

    def sample(self, n: int):  # noqa: ANN001
        return self._df.head(n)


class TestCliInternalHelpers:
    """Tests for internal helper functions used by CLI commands."""

    def test_resolve_out_dir_creates_directory(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "output"

        resolved = cli_main._resolve_out_dir(out_dir)

        assert resolved == out_dir
        assert resolved.exists()
        assert resolved.is_dir()

    def test_resolve_out_dir_rejects_file_path(self, tmp_path: Path) -> None:
        invalid_path = tmp_path / "not_a_directory"
        invalid_path.write_text("file", encoding="utf-8")

        with pytest.raises(typer.BadParameter, match="not a directory"):
            cli_main._resolve_out_dir(invalid_path)

    def test_resolve_out_dir_defaults_to_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)

        resolved = cli_main._resolve_out_dir(None)

        assert resolved == Path("output")
        assert (tmp_path / "output").is_dir()

    def test_load_real_dataframe_or_exit_wraps_loader_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / "real.csv"
        csv_path.write_text("amount\n100\n", encoding="utf-8")

        class FailingCSVLoader:
            def load(self, path: Path):  # noqa: ANN001
                raise ValueError("boom")

        monkeypatch.setattr(cli_main, "CSVLoader", FailingCSVLoader)

        with pytest.raises(typer.BadParameter, match="Failed to load real data: boom"):
            cli_main._load_real_dataframe_or_exit(
                csv_path,
                error_prefix="Failed to load real data",
            )
