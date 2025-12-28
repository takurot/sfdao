import io
from pathlib import Path

from rich.console import Console

from sfdao.cli.audit import run_audit


def test_run_audit_writes_html_report(tmp_path: Path) -> None:
    real_csv = tmp_path / "real.csv"
    synthetic_csv = tmp_path / "synthetic.csv"
    output_path = tmp_path / "report.html"

    real_csv.write_text("amount,balance\n1.0,10.0\n2.0,12.0\n3.0,13.0\n")
    synthetic_csv.write_text("amount,balance\n1.1,10.1\n2.1,12.2\n3.1,13.3\n")

    console = Console(file=io.StringIO())
    run_audit(
        real_path=real_csv,
        synthetic_path=synthetic_csv,
        output_path=output_path,
        quiet=True,
        console=console,
    )

    assert output_path.exists()
    content = output_path.read_text()
    assert "<html" in content.lower()
    assert "Overall Score" in content
