"""Utilities for generating simple synthetic credit card data for testing."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["generate_simple_synthetic"]


def _is_label_column(column_name: str) -> bool:
    """Determine whether a column represents a class/label field."""

    return column_name.lower() == "class"


def _sample_numeric_column(
    real_values: pd.Series, rng: np.random.Generator, n_samples: int
) -> Iterable[float]:
    """Sample numeric values preserving mean and standard deviation."""

    mean = real_values.mean()
    std = real_values.std()
    adjusted_std = std if std > 0 else 1e-6
    return rng.normal(mean, adjusted_std, n_samples)


def generate_simple_synthetic(
    real_csv_path: str | Path,
    output_path: str | Path,
    n_samples: int = 10000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Generate a simple synthetic dataset by matching basic statistics from a real CSV.

    Parameters
    ----------
    real_csv_path: str | Path
        Path to the real CSV file.
    output_path: str | Path
        Location to write the generated synthetic CSV.
    n_samples: int
        Number of rows to generate.
    random_state: int | None
        Optional seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        The generated synthetic dataframe.
    """

    real_csv_path = Path(real_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_df = pd.read_csv(real_csv_path)
    rng = np.random.default_rng(random_state)

    synthetic_data: dict[str, Iterable[float]] = {}
    for column in real_df.columns:
        if _is_label_column(column):
            synthetic_data[column] = rng.choice(real_df[column].values, size=n_samples)
        else:
            synthetic_data[column] = _sample_numeric_column(real_df[column], rng, n_samples)

    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv(output_path, index=False)
    return synthetic_df


def _parse_args() -> tuple[Path, Path, int, int | None]:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Generate simple synthetic credit card data")
    parser.add_argument("real_csv_path", type=Path, help="Path to the real CSV dataset")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path where the synthetic CSV will be written",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of synthetic rows to generate (default: 10000)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )

    args = parser.parse_args()
    return args.real_csv_path, args.output_path, args.n_samples, args.random_state


def main() -> None:  # pragma: no cover - thin CLI wrapper
    real_csv_path, output_path, n_samples, random_state = _parse_args()
    generate_simple_synthetic(
        real_csv_path, output_path, n_samples=n_samples, random_state=random_state
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
