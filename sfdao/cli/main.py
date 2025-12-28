"""Main CLI entry point for SFDAO.

This module provides the command-line interface for the Synthetic Finance
Data Auditor & Optimizer tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from sfdao.cli.audit import run_audit

__all__ = ["app"]

app = typer.Typer(
    name="sfdao",
    help="Synthetic Finance Data Auditor & Optimizer - "
    "A tool for evaluating synthetic financial data quality.",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Display version and exit."""
    if value:
        console.print("sfdao version 0.1.0")
        raise typer.Exit()


def validate_file_exists(path: Optional[Path], name: str) -> Path:
    """Validate that a file exists."""
    if path is None:
        raise typer.BadParameter(f"Missing option '--{name}'.")
    if not path.exists():
        raise typer.BadParameter(f"File '{path}' does not exist.")
    if not path.is_file():
        raise typer.BadParameter(f"'{path}' is not a file.")
    return path


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """SFDAO - Synthetic Finance Data Auditor & Optimizer.

    A comprehensive tool for evaluating the quality, fidelity, and privacy
    of synthetic financial data compared to real data.
    """


@app.command()
def audit(
    real: Annotated[
        Optional[Path],
        typer.Option(
            "--real",
            "-r",
            help="Path to the real data CSV file.",
        ),
    ] = None,
    synthetic: Annotated[
        Optional[Path],
        typer.Option(
            "--synthetic",
            "-s",
            help="Path to the synthetic data CSV file.",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Path to save the evaluation report. "
            "If not specified, output is printed to console.",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress console output (only write to file if --output is specified).",
        ),
    ] = False,
) -> None:
    """Run audit evaluation on synthetic data against real data.

    This command compares a synthetic dataset to a real dataset and generates
    a comprehensive quality evaluation report including:

    - Statistical quality metrics (KS test, JS divergence)
    - Financial stylized facts evaluation
    - Privacy risk assessment
    - Composite quality score

    Examples:
        # Basic audit with console output
        sfdao audit --real data/real.csv --synthetic data/synthetic.csv

        # Audit with report file output
        sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.txt
    """
    # Validate required arguments
    real_path = validate_file_exists(real, "real")
    synthetic_path = validate_file_exists(synthetic, "synthetic")

    run_audit(
        real_path=real_path,
        synthetic_path=synthetic_path,
        output_path=output,
        quiet=quiet,
        console=console,
    )


if __name__ == "__main__":
    app()
