import json
import subprocess
import sys

import numpy as np
import pandas as pd


def test_privacy_cli_smoke(tmp_path):
    """Smoke: run privacy evaluator via CLI stub and write JSON output."""

    rng = np.random.default_rng(11)
    real = rng.normal(0, 1, size=(80, 3))
    synthetic = real[:40] + rng.normal(0, 0.05, size=(40, 3))

    real_csv = tmp_path / "real.csv"
    synth_csv = tmp_path / "synthetic.csv"
    out_file = tmp_path / "privacy_result.json"

    pd.DataFrame(real).to_csv(real_csv, index=False)
    pd.DataFrame(synthetic).to_csv(synth_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "sfdao.evaluator.privacy_cli",
            "--real",
            str(real_csv),
            "--synthetic",
            str(synth_csv),
            "--output",
            str(out_file),
        ],
        check=True,
    )

    result = json.loads(out_file.read_text())

    assert "median_dcr" in result
    assert "risk" in result
    assert result["median_dcr"] > 0
    assert 0 <= result["risk"] <= 1
