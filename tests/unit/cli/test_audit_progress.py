"""Unit tests for audit progress UI options."""

from pathlib import Path

from typer.testing import CliRunner

from sfdao.cli.main import app

runner = CliRunner()


def _write_sample_csv(path: Path) -> None:
    path.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")


def test_audit_progress_outputs_steps(tmp_path: Path) -> None:
    real_csv = tmp_path / "real.csv"
    synthetic_csv = tmp_path / "synthetic.csv"
    output_file = tmp_path / "report.txt"

    _write_sample_csv(real_csv)
    _write_sample_csv(synthetic_csv)

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
            "--status-interval",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "Step 1/6" in result.output


def test_audit_no_progress_hides_steps(tmp_path: Path) -> None:
    real_csv = tmp_path / "real.csv"
    synthetic_csv = tmp_path / "synthetic.csv"
    output_file = tmp_path / "report.txt"

    _write_sample_csv(real_csv)
    _write_sample_csv(synthetic_csv)

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
            "--no-progress",
            "--status-interval",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "Step 1/6" not in result.output


def test_audit_verbose_emits_details(tmp_path: Path) -> None:
    real_csv = tmp_path / "real.csv"
    synthetic_csv = tmp_path / "synthetic.csv"
    output_file = tmp_path / "report.txt"

    _write_sample_csv(real_csv)
    _write_sample_csv(synthetic_csv)

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
            "--verbose",
            "--status-interval",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "Shared numeric columns" in result.output
