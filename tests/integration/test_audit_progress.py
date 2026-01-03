"""Integration tests for audit progress reporting."""

from pathlib import Path

from rich.console import Console

from sfdao.cli.audit import run_audit


def _write_sample_csv(path: Path) -> None:
    path.write_text("col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")


def test_run_audit_emits_progress_steps(tmp_path: Path) -> None:
    real_csv = tmp_path / "real.csv"
    synthetic_csv = tmp_path / "synthetic.csv"
    output_file = tmp_path / "report.txt"

    _write_sample_csv(real_csv)
    _write_sample_csv(synthetic_csv)

    console = Console(record=True, force_terminal=True, width=120)

    run_audit(
        real_path=real_csv,
        synthetic_path=synthetic_csv,
        output_path=output_file,
        quiet=False,
        console=console,
        status_interval=0,
        no_progress=False,
        verbose=False,
    )

    assert output_file.exists()
    assert "Step 1/6" in console.export_text()
