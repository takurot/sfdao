"""Main CLI entry point for SFDAO.

This module provides the command-line interface for the Synthetic Finance
Data Auditor & Optimizer tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import ValidationError
from rich.console import Console

from sfdao.cli.audit import run_audit
from sfdao.config.loader import load_phase2_config
from sfdao.config.models import Phase2Config
from sfdao.generator.factory import build_generator
from sfdao.ingestion.loader import CSVLoader

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


def validate_output_path(path: Optional[Path], name: str) -> Path:
    """Validate and prepare an output file path."""
    if path is None:
        raise typer.BadParameter(f"Missing option '--{name}'.")
    if path.exists() and path.is_dir():
        raise typer.BadParameter(f"'{path}' is a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_phase2_config_or_exit(config_path: Path) -> Phase2Config:
    try:
        return load_phase2_config(config_path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


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


@app.command()
def generate(
    real: Annotated[
        Optional[Path],
        typer.Option(
            "--real",
            "-r",
            help="Path to the real data CSV file used to fit the generator.",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to Phase 2 YAML/JSON config file.",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Path to write the generated synthetic CSV file.",
        ),
    ] = None,
    validate_only: Annotated[
        bool,
        typer.Option(
            "--validate-only",
            help="Validate the config file and exit without generating output.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress console output.",
        ),
    ] = False,
) -> None:
    """Generate synthetic data (Phase 2).

    Phase 2 implementation is incremental. In PR#14 this command supports baseline CSV generation.
    """
    config_path = validate_file_exists(config, "config")
    phase2_config = _load_phase2_config_or_exit(config_path)

    if validate_only:
        console.print("[green]✓[/green] Config is valid.")
        return

    real_path = validate_file_exists(real, "real")
    output_path = validate_output_path(output, "output")

    try:
        real_df = CSVLoader().load(real_path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        guard_engine = None
        if phase2_config.guard:
            from sfdao.guard.factory import create_guard_engine

            guard_engine = create_guard_engine(phase2_config.guard)

        generator = build_generator(
            phase2_config.generator, seed=phase2_config.seed, guard=guard_engine
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    generator.fit(real_df)
    synthetic_df = generator.sample(phase2_config.generator.n_samples)
    synthetic_df.to_csv(output_path, index=False)

    if not quiet:
        console.print(f"[green]✓[/green] Wrote synthetic CSV: {output_path}")


@app.command()
def run(
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to Phase 2 YAML/JSON config file.",
        ),
    ] = None,
    validate_only: Annotated[
        bool,
        typer.Option(
            "--validate-only",
            help="Validate the config file and exit without running the pipeline.",
        ),
    ] = False,
) -> None:
    """Run generate → guard → audit pipeline (Phase 2).

    Phase 2 implementation is incremental. In PR#13 this command supports config validation.
    """
    config_path = validate_file_exists(config, "config")
    _load_phase2_config_or_exit(config_path)

    if validate_only:
        console.print("[green]✓[/green] Config is valid.")
        return

    raise typer.BadParameter("Pipeline is not implemented yet. Use --validate-only for now.")


if __name__ == "__main__":
    app()
