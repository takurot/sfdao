from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from sfdao.scripts.generate_test_synthetic_data import generate_simple_synthetic


def test_full_audit_pipeline_generates_html_report(tmp_path: Path) -> None:
    """Run the CLI audit end-to-end and assert an HTML report is produced."""

    real_csv = Path("tests/fixtures/creditcard_real_sample.csv")
    synthetic_csv = tmp_path / "synthetic.csv"
    report_path = tmp_path / "report.html"

    generate_simple_synthetic(real_csv, synthetic_csv, n_samples=50, random_state=42)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sfdao.cli.main",
            "audit",
            "--real",
            str(real_csv),
            "--synthetic",
            str(synthetic_csv),
            "--output",
            str(report_path),
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert report_path.exists()

    html = report_path.read_text(encoding="utf-8")
    assert "SFDAO Audit Report" in html
    assert "Overall Score" in html
    assert "privacy_risk" in html
