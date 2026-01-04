import cProfile
import pstats
import time
from pathlib import Path
import pandas as pd
from rich.console import Console

from sfdao.cli.audit import run_audit
from sfdao.scripts.generate_test_synthetic_data import generate_simple_synthetic


def benchmark_audit(n_rows: int = 200000) -> None:
    console = Console()
    console.print(f"[bold blue]Starting benchmark with {n_rows} rows...[/bold blue]")

    # 1. Setup Data
    base_dir = Path("benchmarks/data")
    base_dir.mkdir(parents=True, exist_ok=True)

    real_source = Path("example/data/creditcard_10000.csv")
    if not real_source.exists():
        console.print("[red]Error: example/data/creditcard.csv not found.[/red]")
        return

    real_bench_path = base_dir / f"real_{n_rows}.csv"
    synthetic_bench_path = base_dir / f"synthetic_{n_rows}.csv"
    report_path = base_dir / f"report_{n_rows}.html"

    # Prepare Real Data (Resample if needed to reach n_rows)
    if not real_bench_path.exists():
        console.print("Preparing real data...")
        df_full = pd.read_csv(real_source)
        if len(df_full) >= n_rows:
            df_real = df_full.head(n_rows)
        else:
            # Resample with replacement to reach n_rows
            df_real = df_full.sample(n=n_rows, replace=True, random_state=42)
        df_real.to_csv(real_bench_path, index=False)

    # Generate Synthetic Data
    if not synthetic_bench_path.exists():
        console.print("Generating synthetic data...")
        generate_simple_synthetic(
            real_csv_path=real_bench_path,
            output_path=synthetic_bench_path,
            n_samples=n_rows,
            random_state=42,
        )

    # 2. Run Benchmark with Profiling
    console.print("Running audit with cProfile...")

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    try:
        run_audit(
            real_path=real_bench_path,
            synthetic_path=synthetic_bench_path,
            output_path=report_path,
            quiet=True,
            console=console,
            no_progress=True,  # Disable progress UI for clean profiling
        )
    except Exception as e:
        console.print(f"[red]Audit failed: {e}[/red]")

    end_time = time.time()
    profiler.disable()

    console.print(
        f"[bold green]Audit completed in {end_time - start_time:.2f} seconds[/bold green]"
    )

    # 3. Save Profile Stats
    stats_path = base_dir / f"audit_{n_rows}.prof"
    stats = pstats.Stats(profiler)
    stats.dump_stats(stats_path)
    console.print(f"Profile stats saved to {stats_path}")

    # Print Top 20 Bottlenecks
    console.print("\n[bold]Top 20 Time-Consuming Functions:[/bold]")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)


if __name__ == "__main__":
    benchmark_audit(200000)
