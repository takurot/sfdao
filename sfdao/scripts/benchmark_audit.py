import argparse
import sys
import time
from pathlib import Path
from rich.console import Console
from sfdao.generator.baseline import BaselineGenerator
from sfdao.ingestion.loader import CSVLoader
from sfdao.cli.audit import run_audit


def main():
    parser = argparse.ArgumentParser(description="Benchmark SFDAO generation and audit.")
    parser.add_argument("--real", required=True, type=Path, help="Path to real data CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save outputs")

    args = parser.parse_args()
    console = Console()

    real_path = args.real
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_path = output_dir / "synthetic_benchmark.csv"
    report_path = output_dir / "audit_report_benchmark.txt"

    console.print(f"[bold blue]Starting Benchmark[/bold blue]")
    console.print(f"Real Data: {real_path}")

    # 1. Load Data
    start_time = time.time()
    loader = CSVLoader()
    try:
        real_df = loader.load(str(real_path))
        load_time = time.time() - start_time
        console.print(f"Data Loading: {load_time:.4f}s ({len(real_df)} rows)")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        sys.exit(1)

    # 2. Generation
    console.print("\n[bold]Benchmarking Generation...[/bold]")
    start_time = time.time()
    generator = BaselineGenerator(seed=42)
    generator.fit(real_df)
    fit_time = time.time() - start_time

    start_time = time.time()
    synthetic_df = generator.sample(len(real_df))
    sample_time = time.time() - start_time

    synthetic_df.to_csv(synthetic_path, index=False)
    total_gen_time = fit_time + sample_time

    console.print(f"Fit Time:    {fit_time:.4f}s")
    console.print(f"Sample Time: {sample_time:.4f}s")
    console.print(f"Total Gen:   {total_gen_time:.4f}s")

    # 3. Audit
    console.print("\n[bold]Benchmarking Audit...[/bold]")
    start_time = time.time()
    try:
        run_audit(
            real_path=real_path,
            synthetic_path=synthetic_path,
            output_path=report_path,
            quiet=True,
            console=console,
        )
        audit_time = time.time() - start_time
        console.print(f"Audit Time:  {audit_time:.4f}s")
    except Exception as e:
        console.print(f"[red]Error during audit: {e}[/red]")
        sys.exit(1)

    console.print("\n[bold green]Benchmark Complete![/bold green]")
    console.print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
