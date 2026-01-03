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
from sfdao.scenario.loader import load_scenario_engine

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
    ml_utility: Annotated[
        bool,
        typer.Option(
            "--ml-utility",
            help="Enable ML utility evaluation (TSTR). Disabled by default due to compute cost.",
        ),
    ] = False,
    ml_target: Annotated[
        Optional[str],
        typer.Option(
            "--ml-target",
            help="Target column for ML utility evaluation (required if --ml-utility is set).",
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
    no_progress: Annotated[
        bool,
        typer.Option(
            "--no-progress",
            help="Disable progress indicators (useful for CI/log-only runs).",
        ),
    ] = False,
    status_interval: Annotated[
        float,
        typer.Option(
            "--status-interval",
            help="Seconds between status updates during long-running phases.",
        ),
    ] = 30.0,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose status output during audit.",
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
    - ML utility evaluation (optional, with --ml-utility)

    Examples:
        # Basic audit with console output
        sfdao audit --real data/real.csv --synthetic data/synthetic.csv

        # Audit with report file output
        sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.txt

        # Audit with ML utility evaluation
        sfdao audit --real data/real.csv --synthetic data/synthetic.csv \\
            --ml-utility --ml-target Class
    """
    # Validate required arguments
    real_path = validate_file_exists(real, "real")
    synthetic_path = validate_file_exists(synthetic, "synthetic")

    # Validate ml_target is provided when ml_utility is enabled
    if ml_utility and not ml_target:
        raise typer.BadParameter("--ml-target is required when --ml-utility is enabled.")

    run_audit(
        real_path=real_path,
        synthetic_path=synthetic_path,
        output_path=output,
        quiet=quiet,
        console=console,
        ml_utility=ml_utility,
        ml_target=ml_target,
        status_interval=status_interval,
        no_progress=no_progress,
        verbose=verbose,
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
    real: Annotated[
        Optional[Path],
        typer.Option(
            "--real",
            "-r",
            help="Path to the real data CSV file.",
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
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            "-o",
            help="Directory to save outputs (synthetic.csv, report.html).",
        ),
    ] = Path("output"),
    validate_only: Annotated[
        bool,
        typer.Option(
            "--validate-only",
            help="Validate the config file and exit without running the pipeline.",
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
    """Run generate → guard → audit pipeline (Phase 2).

    Executes the full pipeline:
    1. Generator: Fit to real data and sample (with Guard/Scenario if configured)
    2. Audit: Evaluate synthetic data against real data
    3. Report: Generate evaluation report
    """
    config_path = validate_file_exists(config, "config")
    phase2_config = _load_phase2_config_or_exit(config_path)

    if validate_only:
        console.print("[green]✓[/green] Config is valid.")
        return

    real_path = validate_file_exists(real, "real")

    # 0. Setup output directory
    if out_dir is None:
        out_dir = Path("output")

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    elif not out_dir.is_dir():
        raise typer.BadParameter(f"'{out_dir}' is not a directory.")

    # 1. Load Real Data
    try:
        real_df = CSVLoader().load(real_path)
    except Exception as e:
        raise typer.BadParameter(f"Failed to load real data: {e}")

    # 2. Build Components
    guard_engine = None
    if phase2_config.guard:
        from sfdao.guard.factory import create_guard_engine

        guard_engine = create_guard_engine(phase2_config.guard)

    scenario_engine = None
    if phase2_config.scenario:
        scenario_engine = load_scenario_engine(phase2_config.scenario, seed=phase2_config.seed)

    try:
        generator = build_generator(
            phase2_config.generator,
            seed=phase2_config.seed,
            guard=guard_engine,
            scenario=scenario_engine,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    # 3. Generate
    if not quiet:
        console.print("[bold blue]Generating synthetic data...[/bold blue]")

    generator.fit(real_df)
    synthetic_df = generator.sample(phase2_config.generator.n_samples)

    synthetic_path = out_dir / "synthetic.csv"
    synthetic_df.to_csv(synthetic_path, index=False)

    if not quiet:
        console.print(f"[green]✓[/green] Synthetic data saved to: {synthetic_path}")

    # 4. Audit
    if not quiet:
        console.print("[bold blue]Running audit...[/bold blue]")

    report_path = out_dir / "report.html"

    weights = phase2_config.audit.weights if phase2_config.audit else None
    privacy_settings = phase2_config.audit.privacy if phase2_config.audit else None

    run_audit(
        real_path=real_path,
        synthetic_path=synthetic_path,
        output_path=report_path,
        quiet=quiet,
        console=console,
        weights=weights,
        privacy_settings=privacy_settings,
    )


if __name__ == "__main__":
    app()
