from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_example_project_smoke_generates_html_report(tmp_path: Path) -> None:
    real_csv = Path("example/data/creditcard_real_sample.csv")
    synthetic_csv = tmp_path / "creditcard_synthetic.csv"
    report_path = tmp_path / "report.html"

    generate_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sfdao.scripts.generate_test_synthetic_data",
            str(real_csv),
            str(synthetic_csv),
            "--n-samples",
            "50",
            "--random-state",
            "42",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        generate_result.returncode == 0
    ), f"stdout:\n{generate_result.stdout}\nstderr:\n{generate_result.stderr}"
    assert synthetic_csv.exists()

    audit_result = subprocess.run(
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
    assert (
        audit_result.returncode == 0
    ), f"stdout:\n{audit_result.stdout}\nstderr:\n{audit_result.stderr}"
    assert report_path.exists()

    html = report_path.read_text(encoding="utf-8")
    assert "SFDAO Audit Report" in html
    assert "Overall Score" in html
