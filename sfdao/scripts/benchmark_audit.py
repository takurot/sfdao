import argparse
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sfdao.generator.baseline import BaselineGenerator
from sfdao.ingestion.loader import CSVLoader
from sfdao.cli.audit import run_audit
from sfdao.config.models import PrivacySettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SFDAO generation and audit.")
    parser.add_argument("--real", required=True, type=Path, help="Path to real data CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save outputs")
    parser.add_argument("--sizes", default="1000,10000", help="Comma-separated sizes to benchmark")
    parser.add_argument(
        "--privacy-sample-size", type=int, help="Sample size for privacy optimization"
    )

    args = parser.parse_args()
    console = Console()

    real_path = args.real
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = [int(s) for s in args.sizes.split(",")]
    console.print(f"[bold blue]Starting Benchmark[/bold blue]")
    console.print(f"Sizes: {sizes}")
    console.print(f"Privacy Sample Size: {args.privacy_sample_size}")

    # 1. Load Data
    start_time = time.time()
    loader = CSVLoader()
    try:
        real_df_full = loader.load(str(real_path))
        load_time = time.time() - start_time
        console.print(f"Data Loading: {load_time:.4f}s ({len(real_df_full)} rows total)")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        sys.exit(1)

    results = []

    for size in sizes:
        console.print(f"\n[bold]Benchmarking Size: {size}[/bold]")

        # Prepare Data
        if len(real_df_full) >= size:
            real_df = real_df_full.sample(n=size, random_state=42)
        else:
            console.print(
                f"[yellow]Warning: Requested size {size} > real data size {len(real_df_full)}. Using full data.[/yellow]"
            )
            real_df = real_df_full.copy()
            # If we can't reach the size for real data, we still generate 'size' rows for synthetic?
            # Usually strict benchmark implies real=size, synthetic=size.
            # But if real is small, we can override size to match real for fairness, OR upscale synthetic.
            # Let's upscale synthetic to requested size, but keep real as max available.
            
        current_real_path = output_dir / f"real_subset_{size}.csv"
        real_df.to_csv(current_real_path, index=False)

        synthetic_path = output_dir / f"synthetic_{size}.csv"
        report_path = output_dir / f"report_{size}.txt"

        # 2. Generation
        start_time = time.time()
        generator = BaselineGenerator(seed=42)
        generator.fit(real_df)
        fit_time = time.time() - start_time

        sample_start = time.time()
        # Generate requested size
        synthetic_df = generator.sample(size)
        sample_time = time.time() - sample_start
        
        synthetic_df.to_csv(synthetic_path, index=False)
        total_gen_time = fit_time + sample_time

        console.print(f"Gen Time:    {total_gen_time:.4f}s (Fit: {fit_time:.2f}s, Sample: {sample_time:.2f}s)")

        # 3. Audit
        privacy_settings = None
        if args.privacy_sample_size:
            privacy_settings = PrivacySettings(sample_size=args.privacy_sample_size)

        start_time = time.time()
        try:
            run_audit(
                real_path=current_real_path,
                synthetic_path=synthetic_path,
                output_path=report_path,
                quiet=True,
                console=console,
                privacy_settings=privacy_settings,
            )
            audit_time = time.time() - start_time
            console.print(f"Audit Time:  {audit_time:.4f}s")
        except Exception as e:
            console.print(f"[red]Error during audit: {e}[/red]")
            audit_time = -1.0

        results.append(
            {
                "Size": str(size),
                "Gen (s)": f"{total_gen_time:.4f}",
                "Audit (s)": f"{audit_time:.4f}",
            }
        )

    # Summary Table
    console.print("\n[bold]Benchmark Results[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Size")
    table.add_column("Gen (s)")
    table.add_column("Audit (s)")

    for res in results:
        table.add_row(res["Size"], res["Gen (s)"], res["Audit (s)"])
    
    console.print(table)
    console.print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
