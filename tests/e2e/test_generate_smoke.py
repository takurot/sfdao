from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_generate_command_writes_csv(tmp_path: Path) -> None:
    real_csv = Path("tests/fixtures/sample_transactions.csv")
    config_path = tmp_path / "phase2.yaml"
    output_path = tmp_path / "synthetic.csv"

    config_path.write_text(
        "\n".join(
            [
                "version: 2",
                "seed: 123",
                "generator:",
                "  type: baseline",
                "  n_samples: 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sfdao.cli.main",
            "generate",
            "--real",
            str(real_csv),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert output_path.exists()
