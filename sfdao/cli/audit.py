"""Audit command implementation for SFDAO CLI.

This module contains the core audit logic that evaluates synthetic data
against real data using the SFDAO evaluator modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.console import Console

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from sfdao.cli.progress import AuditProgress, AuditProgressConfig
from sfdao.config.models import PrivacySettings
from sfdao.evaluator.scoring import CompositeScorer
from sfdao.evaluator.financial_facts import FinancialFactsChecker
from sfdao.evaluator.ml_utility import MLUtilityEvaluator, MLUtilityResult
from sfdao.evaluator.privacy import PrivacyEvaluator
from sfdao.evaluator.statistical import StatisticalEvaluator
from sfdao.ingestion.loader import CSVLoader
from sfdao.reporter.base import EvaluationReport, PlainTextReporter
from sfdao.reporter.html import HTMLReporter
from sfdao.reporter.pdf import PDFReporter

__all__ = ["run_audit"]


def run_audit(
    real_path: Path,
    synthetic_path: Path,
    output_path: Optional[Path],
    quiet: bool,
    console: Console,
    weights: Optional[dict[str, float]] = None,
    privacy_settings: Optional[PrivacySettings] = None,
    ml_utility: bool = False,
    ml_target: Optional[str] = None,
    status_interval: float = 30.0,
    no_progress: bool = False,
    verbose: bool = False,
) -> None:
    """Run audit evaluation and generate report.

    Args:
        real_path: Path to the real data CSV file.
        synthetic_path: Path to the synthetic data CSV file.
        output_path: Optional path to save the report.
        quiet: If True, suppress console output.
        console: Rich console for output.
        weights: Optional dictionary of component weights.
        privacy_settings: Optional privacy evaluation settings.
        ml_utility: If True, run ML utility (TSTR) evaluation.
        ml_target: Target column for ML utility evaluation.
    """
    progress = AuditProgress(
        console,
        AuditProgressConfig(
            total_steps=6,
            quiet=quiet,
            no_progress=no_progress,
            status_interval=status_interval,
            verbose=verbose,
        ),
    )

    if not quiet:
        console.print("[bold blue]SFDAO Audit[/bold blue] - Starting evaluation...")

    # Load data
    loader = CSVLoader()
    with progress.start_phase("Load data"):
        real_df = loader.load(str(real_path))
        synthetic_df = loader.load(str(synthetic_path))

    if not quiet:
        console.print(f"  Real data: {len(real_df)} rows, {len(real_df.columns)} columns")
        console.print(
            f"  Synthetic data: {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns"
        )

    # Evaluate statistical quality for numeric columns
    statistical_evaluator = StatisticalEvaluator()
    metrics: dict[str, float] = {}

    numeric_columns = real_df.select_dtypes(include=["number"]).columns.tolist()
    shared_numeric = [col for col in numeric_columns if col in synthetic_df.columns]
    progress.log_verbose(f"Shared numeric columns: {len(shared_numeric)}")

    with progress.start_phase(
        "Statistical evaluation",
        detail=f"{len(shared_numeric)} numeric columns" if shared_numeric else "no numeric overlap",
    ):
        if shared_numeric:
            # Calculate average KS statistic across all numeric columns
            ks_statistics = []
            js_divergences = []

            with progress.track(shared_numeric, "  Computing statistical metrics") as columns:
                for col in columns:
                    real_values = real_df[col].dropna().values
                    synthetic_values = synthetic_df[col].dropna().values

                    if len(real_values) > 0 and len(synthetic_values) > 0:
                        # KS test
                        ks_result = statistical_evaluator.ks_test(real_values, synthetic_values)
                        ks_statistics.append(ks_result.statistic)

                        # JS divergence
                        js_result = statistical_evaluator.js_divergence(
                            real_values, synthetic_values
                        )
                        js_divergences.append(js_result)

            # Convert to quality scores (1 - statistic, higher is better)
            if ks_statistics:
                avg_ks = sum(ks_statistics) / len(ks_statistics)
                metrics["quality"] = max(0.0, 1.0 - avg_ks)
            else:
                metrics["quality"] = 0.5

            if js_divergences:
                avg_js = sum(js_divergences) / len(js_divergences)
                metrics["utility"] = max(0.0, 1.0 - avg_js)
            else:
                metrics["utility"] = 0.5
        else:
            metrics["quality"] = 0.5
            metrics["utility"] = 0.5

    # Privacy evaluation
    # We estimate total steps for progress bar if possible (based on synthetic rows)
    # But PrivacyEvaluator manages batching.
    # We will pass a progress hook.

    with progress.start_phase(
        "Privacy evaluation",
        detail=f"sample={privacy_settings.sample_size}" if privacy_settings else None,
    ):
        # We want to show a progress bar for the DCR calculation
        # This requires accessing the progress context.
        # Since AuditProgress wraps phases, let's pass the progress object or a callback adapter.

        # We will create a temporary progress task for this specific long-running op
        # inside _compute_privacy_scores if we want a bar.
        # However, `start_phase` returns a Heartbeat (spinner/text).
        # A progress bar would conflict with the spinner on the same line if not careful.
        # But `start_phase` is a context manager that clears itself (or persists).

        # Let's delegate to _compute_privacy_scores and pass the console/progress config
        privacy_score, privacy_risk, privacy_dcr_median = _compute_privacy_scores(
            real_df, synthetic_df, shared_numeric, privacy_settings, console, quiet, no_progress
        )
        metrics["privacy"] = privacy_score

    with progress.start_phase("Financial facts"):
        financial_facts = _compute_financial_facts(real_df, synthetic_df, shared_numeric)

    if ml_utility and ml_target:
        with progress.start_phase("ML utility evaluation"):
            if not quiet:
                console.print("  Computing ML utility (TSTR)...")
            ml_result = _compute_ml_utility(real_df, synthetic_df, ml_target, quiet, console)
    else:
        with progress.start_phase("ML utility evaluation", detail="disabled"):
            ml_result = None

    with progress.start_phase("Report output"):
        if not quiet:
            console.print("  Calculating composite score...")
        # Calculate composite score
        if weights is None:
            weights = {"quality": 0.4, "utility": 0.3, "privacy": 0.3}
        scorer = CompositeScorer(weights)
        composite_score = scorer.calculate(metrics)

        # Create evaluation report
        metadata: dict[str, object] = {
            "real_file": str(real_path),
            "synthetic_file": str(synthetic_path),
            "real_rows": len(real_df),
            "synthetic_rows": len(synthetic_df),
            "privacy_risk": privacy_risk,
            "privacy_dcr_median": privacy_dcr_median,
            "financial_facts": financial_facts,
        }
        if privacy_settings and privacy_settings.sample_size:
            metadata["privacy_sample_size"] = privacy_settings.sample_size

        # ML Utility evaluation (optional)
        if ml_result is not None:
            metadata["ml_utility"] = {
                "tstr_auc": ml_result.tstr_auc,
                "tstr_f1": ml_result.tstr_f1,
                "trtr_auc": ml_result.trtr_auc,
                "trtr_f1": ml_result.trtr_f1,
                "utility_ratio": ml_result.utility_ratio,
                "model_type": ml_result.model_type,
                "target_column": ml_result.target_column,
                "n_features": ml_result.n_features,
            }

        report = EvaluationReport(
            metrics=metrics,
            composite_score=composite_score,
            metadata=metadata,
        )

        reporter = _select_reporter(output_path)

        # Output report
        if output_path:
            reporter.render_to_file(report, output_path)
            if not quiet:
                console.print(f"[green]✓[/green] Report saved to: {output_path}")
        else:
            report_text = reporter.generate(report)
            if not quiet:
                console.print("\n[bold]Evaluation Report:[/bold]")
                if isinstance(report_text, bytes):
                    console.print(
                        "[yellow]Binary report generated. Use --output to save to a file.[/yellow]"
                    )
                else:
                    console.print(report_text)

        if not quiet:
            console.print(
                f"\n[bold green]Audit complete![/bold green] Overall Score: "
                f"{composite_score.total:.3f}"
            )


def _compute_privacy_scores(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    shared_numeric: list[str],
    privacy_settings: Optional[PrivacySettings] = None,
    console: Console | None = None,
    quiet: bool = False,
    no_progress: bool = False,
) -> tuple[float, float | None, float | None]:
    if not shared_numeric:
        return 0.5, None, None

    real_numeric = real_df[shared_numeric].dropna()
    synthetic_numeric = synthetic_df[shared_numeric].dropna()

    if real_numeric.empty or synthetic_numeric.empty:
        return 0.5, None, None

    sample_size = privacy_settings.sample_size if privacy_settings else None
    evaluator = PrivacyEvaluator(sample_size=sample_size)
    real_matrix = real_numeric.to_numpy(dtype=float)
    synthetic_matrix = synthetic_numeric.to_numpy(dtype=float)

    # Progress Bar Setup
    progress_callback = None
    if console and not quiet and not no_progress:
        # Use a temporary Progress instance for the calculation
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        )

        task_id = progress.add_task("  Computing risk...", total=len(synthetic_matrix))
        progress.start()

        def _update(n: int) -> None:
            progress.advance(task_id, advance=n)

        progress_callback = _update

    try:
        # Calculate DCR once with progress, then compute risk manually (avoids 2x calc)
        dcr = evaluator.distance_to_closest_record(
            real_matrix,
            synthetic_matrix,
            progress_callback=progress_callback,
        )

        # Calculate risk using the DCR we just computed
        scale = evaluator.reference_distance(real_matrix)
        scaled = np.exp(-dcr / (scale + 1e-12))
        clipped = np.clip(scaled, 0.0, 1.0)
        risk = float(np.mean(clipped))

    finally:
        if progress_callback and "progress" in locals():
            progress.stop()

    dcr_median = float(np.median(dcr)) if dcr.size > 0 else None

    privacy_score = max(0.0, min(1.0, 1.0 - risk))
    return privacy_score, risk, dcr_median


def _compute_financial_facts(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    shared_numeric: list[str],
) -> dict[str, dict[str, dict[str, float | int | str]]]:
    if not shared_numeric:
        return {}

    checker = FinancialFactsChecker()
    results: dict[str, dict[str, dict[str, float | int | str]]] = {}

    for col in shared_numeric:
        real_values: NDArray[np.float64] = real_df[col].dropna().to_numpy(dtype=float)
        synthetic_values: NDArray[np.float64] = synthetic_df[col].dropna().to_numpy(dtype=float)

        if len(real_values) == 0 or len(synthetic_values) == 0:
            continue

        results[col] = {
            "real": _summarize_financial_facts(checker, real_values),
            "synthetic": _summarize_financial_facts(checker, synthetic_values),
        }

    return results


def _summarize_financial_facts(
    checker: FinancialFactsChecker,
    values: NDArray[np.float64],
) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {}

    fat_tail = checker.check_fat_tail(values)
    summary["fat_tail_kurtosis"] = fat_tail.kurtosis
    summary["fat_tail_excess_kurtosis"] = fat_tail.excess_kurtosis
    summary["fat_tail_sample_size"] = fat_tail.sample_size

    if len(values) >= 11:
        volatility = checker.check_volatility_clustering(values, lags=10)
        summary["volatility_ljung_box_stat"] = volatility.ljung_box_statistic
        summary["volatility_ljung_box_p_value"] = volatility.ljung_box_p_value
        summary["volatility_arch_stat"] = volatility.arch_test_statistic
        summary["volatility_arch_p_value"] = volatility.arch_test_p_value
        summary["volatility_lags"] = volatility.lags
    else:
        summary["volatility_note"] = "insufficient data for volatility clustering"

    return summary


def _select_reporter(output_path: Optional[Path]) -> PlainTextReporter | HTMLReporter | PDFReporter:
    if output_path is None:
        return PlainTextReporter()

    suffix = output_path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return HTMLReporter()
    if suffix == ".pdf":
        return PDFReporter()
    return PlainTextReporter()


def _compute_ml_utility(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_column: str,
    quiet: bool,
    console: Console,
) -> Optional[MLUtilityResult]:
    """Compute ML utility (TSTR) evaluation.

    Args:
        real_df: Real data DataFrame.
        synthetic_df: Synthetic data DataFrame.
        target_column: Target column for classification.
        quiet: If True, suppress console output.
        console: Rich console for output.

    Returns:
        MLUtilityResult if successful, None if evaluation fails.
    """
    try:
        evaluator = MLUtilityEvaluator()
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=synthetic_df,
            target_column=target_column,
        )
        if not quiet:
            console.print(
                f"    TSTR AUC: {result.tstr_auc:.3f}, "
                f"TRTR AUC: {result.trtr_auc:.3f}, "
                f"Utility Ratio: {result.utility_ratio:.3f}"
            )
        return result
    except ValueError as e:
        if not quiet:
            console.print(f"[yellow]Warning: ML utility evaluation failed: {e}[/yellow]")
        return None
