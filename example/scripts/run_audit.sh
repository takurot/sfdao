#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

poetry run sfdao audit \
  --real example/data/creditcard_real_sample.csv \
  --synthetic example/output/creditcard_synthetic.csv \
  --output example/output/report.html
