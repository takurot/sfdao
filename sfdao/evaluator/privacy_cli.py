from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sfdao.evaluator.privacy import PrivacyEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run privacy evaluation on CSV files.")
    parser.add_argument("--real", required=True, type=Path, help="Path to real data CSV.")
    parser.add_argument(
        "--synthetic", required=True, type=Path, help="Path to synthetic data CSV."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output JSON summary (median DCR and risk).",
    )
    return parser.parse_args()


def load_numeric_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError(f"No numeric columns found in {path}")
    return numeric_df.to_numpy(dtype=float)


def main() -> None:
    args = parse_args()
    real = load_numeric_matrix(args.real)
    synthetic = load_numeric_matrix(args.synthetic)

    evaluator = PrivacyEvaluator()
    dcr = evaluator.distance_to_closest_record(real, synthetic)
    risk = evaluator.reidentification_risk(real, synthetic)

    summary = {"median_dcr": float(np.median(dcr)), "risk": risk}
    args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
