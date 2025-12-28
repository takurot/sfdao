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

from sfdao.evaluator.scoring import CompositeScorer
from sfdao.evaluator.financial_facts import FinancialFactsChecker
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
) -> None:
    """Run audit evaluation and generate report.

    Args:
        real_path: Path to the real data CSV file.
        synthetic_path: Path to the synthetic data CSV file.
        output_path: Optional path to save the report.
        quiet: If True, suppress console output.
        console: Rich console for output.
    """
    if not quiet:
        console.print("[bold blue]SFDAO Audit[/bold blue] - Starting evaluation...")

    # Load data
    loader = CSVLoader()
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

    if shared_numeric:
        # Calculate average KS statistic across all numeric columns
        ks_statistics = []
        js_divergences = []

        for col in shared_numeric:
            if col in synthetic_df.columns:
                real_values = real_df[col].dropna().values
                synthetic_values = synthetic_df[col].dropna().values

                if len(real_values) > 0 and len(synthetic_values) > 0:
                    # KS test
                    ks_result = statistical_evaluator.ks_test(real_values, synthetic_values)
                    ks_statistics.append(ks_result.statistic)

                    # JS divergence
                    js_result = statistical_evaluator.js_divergence(real_values, synthetic_values)
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

    privacy_score, privacy_risk, privacy_dcr_median = _compute_privacy_scores(
        real_df, synthetic_df, shared_numeric
    )
    metrics["privacy"] = privacy_score

    if not quiet:
        console.print("  Calculating composite score...")

    # Calculate composite score
    weights = {"quality": 0.4, "utility": 0.3, "privacy": 0.3}
    scorer = CompositeScorer(weights)
    composite_score = scorer.calculate(metrics)

    financial_facts = _compute_financial_facts(real_df, synthetic_df, shared_numeric)

    # Create evaluation report
    report = EvaluationReport(
        metrics=metrics,
        composite_score=composite_score,
        metadata={
            "real_file": str(real_path),
            "synthetic_file": str(synthetic_path),
            "real_rows": len(real_df),
            "synthetic_rows": len(synthetic_df),
            "privacy_risk": privacy_risk,
            "privacy_dcr_median": privacy_dcr_median,
            "financial_facts": financial_facts,
        },
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
            f"\n[bold green]Audit complete![/bold green] Overall Score: {composite_score.total:.3f}"
        )


def _compute_privacy_scores(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    shared_numeric: list[str],
) -> tuple[float, float | None, float | None]:
    if not shared_numeric:
        return 0.5, None, None

    real_numeric = real_df[shared_numeric].dropna()
    synthetic_numeric = synthetic_df[shared_numeric].dropna()

    if real_numeric.empty or synthetic_numeric.empty:
        return 0.5, None, None

    evaluator = PrivacyEvaluator()
    real_matrix = real_numeric.to_numpy(dtype=float)
    synthetic_matrix = synthetic_numeric.to_numpy(dtype=float)

    risk = evaluator.reidentification_risk(real_matrix, synthetic_matrix)
    dcr = evaluator.distance_to_closest_record(real_matrix, synthetic_matrix)
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
