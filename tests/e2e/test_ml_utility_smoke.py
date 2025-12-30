"""E2E smoke test for ML Utility evaluation.

This test verifies the ML utility evaluation works end-to-end
through the CLI audit command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data_with_target(tmp_path: Path) -> tuple[Path, Path]:
    """Create sample real and synthetic data with a classification target."""
    np.random.seed(42)
    n_samples = 200

    # Generate linearly separable data
    X_real = np.random.randn(n_samples, 3)
    y_real = (X_real[:, 0] + X_real[:, 1] > 0).astype(int)
    real_df = pd.DataFrame(X_real, columns=["f1", "f2", "f3"])
    real_df["Class"] = y_real

    # Generate synthetic data from similar distribution
    X_synthetic = np.random.randn(n_samples, 3)
    y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] > 0).astype(int)
    synthetic_df = pd.DataFrame(X_synthetic, columns=["f1", "f2", "f3"])
    synthetic_df["Class"] = y_synthetic

    real_path = tmp_path / "real.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    real_df.to_csv(real_path, index=False)
    synthetic_df.to_csv(synthetic_path, index=False)

    return real_path, synthetic_path


class TestMLUtilitySmokeTest:
    """E2E smoke tests for ML utility evaluation."""

    def test_ml_utility_cli_with_html_output(
        self, sample_data_with_target: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Test ML utility evaluation through CLI produces valid HTML report."""
        real_path, synthetic_path = sample_data_with_target
        output_path = tmp_path / "report.html"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sfdao.cli.main",
                "audit",
                "--real",
                str(real_path),
                "--synthetic",
                str(synthetic_path),
                "--ml-utility",
                "--ml-target",
                "Class",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}\n{result.stdout}"
        assert output_path.exists()

        html_content = output_path.read_text()
        assert "ML Utility" in html_content
        assert "TSTR AUC" in html_content
        assert "TRTR AUC" in html_content
        assert "Utility Ratio" in html_content

    def test_ml_utility_cli_console_output(
        self, sample_data_with_target: tuple[Path, Path]
    ) -> None:
        """Test ML utility evaluation console output."""
        real_path, synthetic_path = sample_data_with_target

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sfdao.cli.main",
                "audit",
                "--real",
                str(real_path),
                "--synthetic",
                str(synthetic_path),
                "--ml-utility",
                "--ml-target",
                "Class",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}\n{result.stdout}"
        # Check for keywords that appear in ML utility output
        # Note: console might wrap lines so we check for individual terms
        # Remove newlines and normalize whitespace to handle line wrapping
        import re

        output_normalized = re.sub(r"\s+", " ", result.stdout)
        assert "TSTR AUC" in output_normalized
        assert "TRTR AUC" in output_normalized
        assert "Utility Ratio" in output_normalized
        assert "Computing ML utility" in output_normalized or "ml_utility" in output_normalized

    def test_ml_utility_requires_target(self, sample_data_with_target: tuple[Path, Path]) -> None:
        """Test that --ml-utility requires --ml-target."""
        real_path, synthetic_path = sample_data_with_target

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sfdao.cli.main",
                "audit",
                "--real",
                str(real_path),
                "--synthetic",
                str(synthetic_path),
                "--ml-utility",
                # No --ml-target
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0

        # Combine stdout and stderr
        combined_output = result.stderr + result.stdout

        # Remove RuntimeWarnings which might clutter CI output
        combined_output = "\n".join(
            line for line in combined_output.splitlines() if "RuntimeWarning" not in line
        )

        # Strip ANSI codes to handle Rich formatting
        import re

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        plain_output = ansi_escape.sub("", combined_output)

        # Check for key parts of the error message
        assert "ml-target" in plain_output
        assert "required" in plain_output

    def test_audit_without_ml_utility(
        self, sample_data_with_target: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Test that audit without --ml-utility does not include ML section."""
        real_path, synthetic_path = sample_data_with_target
        output_path = tmp_path / "report.html"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sfdao.cli.main",
                "audit",
                "--real",
                str(real_path),
                "--synthetic",
                str(synthetic_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}\n{result.stdout}"
        assert output_path.exists()

        html_content = output_path.read_text()
        # ML Utility section should NOT be present when --ml-utility is not used
        assert "ML Utility (TSTR)" not in html_content
