import sys
from pathlib import Path
import subprocess
import yaml

from sfdao.ingestion.loader import CSVLoader


def test_run_pipeline_smoke(tmp_path: Path):
    """Test the full 'sfdao run' pipeline with small data."""
    # 1. Prepare data
    real_csv = tmp_path / "real.csv"
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        {
            "amount": np.random.rand(50) * 1000,
            "category": np.random.choice(["A", "B", "C"], 50),
            "label": [0] * 45 + [1] * 5,
        }
    )
    df.to_csv(real_csv, index=False)

    # 2. Prepare Config in a file
    config_data = {
        "version": 2,
        "seed": 42,
        "generator": {"type": "baseline", "n_samples": 20},
        "audit": {"weights": {"quality": 0.5, "utility": 0.5, "privacy": 0.0}},
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # 3. Define output directory
    out_dir = tmp_path / "output"

    # 4. Run 'sfdao run'
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sfdao.cli.main",
            "run",
            "--real",
            str(real_csv),
            "--config",
            str(config_file),
            "--out-dir",
            str(out_dir),
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )

    # 5. Assertions
    assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    # Check outputs
    synthetic_csv = out_dir / "synthetic.csv"
    report_html = out_dir / "report.html"

    assert synthetic_csv.exists(), "Synthetic data was not created"
    assert report_html.exists(), "Audit report was not created"

    # Verify synthetic data size
    loader = CSVLoader()
    df_syn = loader.load(synthetic_csv)
    assert len(df_syn) == 20
