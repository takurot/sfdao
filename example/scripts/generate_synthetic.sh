#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

poetry run python -m sfdao.scripts.generate_test_synthetic_data \
  example/data/creditcard_real_sample.csv \
  example/output/creditcard_synthetic.csv \
  --n-samples 200 \
  --random-state 42
