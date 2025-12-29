# SFDAO Benchmark Guide

This document describes how to run performance benchmarks for `sfdao`.

## Overview

The benchmarking script (`sfdao/scripts/benchmark_audit.py`) measures the performance of:
1. **Generation** (Fit & Sample time)
2. **Audit** (Evaluation statistics, Privacy risk calculation)

It supports running tests on multiple dataset sizes to evaluate scalability.

## Prerequisites

- Real dataset (CSV format). We recommend the Kaggle Credit Card Fraud Detection dataset or the provided `tests/fixtures/creditcard_sample_10k.csv` for smaller tests.
- Python environment with `sfdao` installed (`poetry install`).

## Usage

Run the benchmark script using `poetry run python`:

```bash
poetry run python sfdao/scripts/benchmark_audit.py \
  --real path/to/real_data.csv \
  --output-dir benchmark_results \
  --sizes 1000,10000,100000
```

### Arguments

- `--real`: Path to the input real data CSV file (required).
- `--output-dir`: Directory where synthetic data, reports, and subset real data will be saved (required).
- `--sizes`: Comma-separated list of row counts to benchmark (default: "1000,10000").
- `--privacy-sample-size`: (Optional) Limit the sample size used for privacy risk calculation (O(N^2)). Use this for large datasets (>10k rows) to reduce audit time while getting a risk estimate.

### Example: Scalability Test with Privacy Optimization

To test scalability up to 100k rows, while keeping privacy check fast (using only 2000 samples for risk calc):

```bash
poetry run python sfdao/scripts/benchmark_audit.py \
  --real data/creditcard_real.csv \
  --output-dir bench_output \
  --sizes 1000,10000,100000 \
  --privacy-sample-size 2000
```

## Interpreting Results

The script outputs a table with:
- **Size**: Number of rows generated/audited.
- **Gen (s)**: Time taken to fit and generate data.
- **Audit (s)**: Time taken to run the full audit suite.

Audit time usually grows quadratically (O(N^2)) due to Privacy Metrics (Distance to Closest Record) if `--privacy-sample-size` is not used. Using `--privacy-sample-size` makes it roughly constant or O(sample_size^2) specifically for the privacy component, though other statistical checks are linear O(N).
