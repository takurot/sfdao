# SFDAO - Synthetic Finance Data Auditor & Optimizer

**Financial Compliance & Synthetic Data Quality Assurance Platform**

[![PyPI version](https://badge.fury.io/py/sfdao.svg)](https://badge.fury.io/py/sfdao)
[![Python Version](https://img.shields.io/pypi/pyversions/sfdao.svg)](https://pypi.org/project/sfdao/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/takurot/sfdao/actions/workflows/ci.yml/badge.svg)](https://github.com/takurot/sfdao/actions)
[![Codecov](https://codecov.io/gh/takurot/sfdao/branch/main/graph/badge.svg)](https://codecov.io/gh/takurot/sfdao)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[日本語版 (Japanese)](README.ja.md)

## Overview

SFDAO is an integrated tool for synthetic data generation, constraint application, and auditing specifically designed for the financial industry. Covering Phases 1 to 3, it handles not only auditing but also generation, guardrail checking, scenario injection, and ML Utility evaluation.

### Key Features

- **Statistical Quality Evaluation**: Distribution comparison using KS test and Jensen-Shannon Divergence.
- **Finance-Specific Evaluation**: Fat Tail detection, Volatility Clustering verification.
- **Privacy Evaluation**: Re-identification risk, Distance to Closest Record (DCR).
- **Automated Type Detection**: Automatic classification of Numeric, Categorical, Datetime, and PII (Personally Identifiable Information).
- **Generation Workflow**: Batch execution of synthetic data generation and auditing via `generate`/`run`.
- **Constraints & Scenarios**: Guardrail rule application, scenario injection (scale/shift/clip/outlier, etc.).
- **ML Utility Evaluation**: Model performance assessment using TSTR (AUC/F1) (optional).
- **Report Generation**: Detailed reports in HTML/PDF formats.

## Installation

### Quick Install (PyPI)

```bash
# Install via pip
pip install sfdao

# Or use pipx for isolated installation (recommended)
pipx install sfdao

# With optional deep learning support (CTGAN)
pip install sfdao[deep]
```

### Prerequisites

- Python 3.10 - 3.12
- macOS: WeasyPrint dependencies for PDF generation
  ```bash
  brew install cairo pango gdk-pixbuf libffi
  ```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/takurot/sfdao.git
cd sfdao

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Enable virtual environment
poetry shell
```

## Quick Start

```bash
# Run a basic audit
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.html

# Output format is automatically detected by extension (.txt/.html/.pdf)
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.txt
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.pdf

# Generate simple synthetic data for testing
poetry run python -m sfdao.scripts.generate_test_synthetic_data \
  example/data/creditcard_real_sample.csv \
  example/output/creditcard_synthetic.csv \
  --n-samples 500 \
  --random-state 42

# Audit the generated synthetic data
poetry run sfdao audit \
  --real example/data/creditcard_real_sample.csv \
  --synthetic example/output/creditcard_synthetic.csv \
  --output example/output/report.html

# Phase 2: Batch execution of Generation -> Guardrails -> Audit
poetry run sfdao run --config example/config/phase2.yaml --outdir example/output
```

## Development

### TDD (Test-Driven Development)

This project is developed using TDD. Follow this cycle when adding new features:

1. **Red**: Write a failing test.
2. **Green**: Write the minimum code to pass the test.
3. **Refactor**: Clean up and optimize the code.

### Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=sfdao --cov-report=html

# Run specific test file
pytest tests/unit/ingestion/test_loader.py
```

### Code Quality

```bash
# Check formatting
black --check .

# Apply formatting
black .

# Lint check
flake8 .

# Type check
mypy sfdao

# Security check
bandit -r sfdao
```

## Project Structure

```
sfdao/
├── sfdao/                  # Main package
│   ├── ingestion/          # Data ingestion and type detection
│   ├── config/             # Configuration schema/loader
│   ├── generator/          # Synthetic data generation
│   ├── guard/              # Rule-based constraint checking
│   ├── scenario/           # Scenario injection
│   ├── evaluator/          # Metric calculation
│   ├── reporter/           # Report generation
│   └── cli/                # CLI interface
├── tests/                  # Test code
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-End tests
├── docs/                   # Documentation
└── prompt/                 # Specifications
```

## Documentation

- [Implementation Plan](prompt/PLAN.md)
- [Product Specifications](prompt/SPEC.md)
- [Example](example/README.md)
- [Usage Guide](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Python API](docs/API.md)
- [Metrics Guide](docs/METRICS.md)

## Roadmap

### Phase 1: "The Auditor" (MVP)

- [x] Project structure and CI/CD setup
- [x] Basic Data Ingestion features
- [x] Auto-Type Detection
- [x] Finance Domain definitions
- [x] Basic Evaluator (statistical tests)
- [x] Financial Stylized Facts evaluation
- [x] Privacy evaluation
- [x] Evaluation scoring integration
- [x] CLI interface
- [x] Report generation feature
- [x] Integration testing and documentation

### Phase 2: "The Generator & Logic"

- [x] Config schema/loader and CLI integration (`generate`/`run`)
- [x] Baseline Generator (statistical sampling)
- [x] Constraint & Logic Guard (rule detection/exclusion/correction)
- [x] Scenario Injection (scale/shift/clip/outlier, etc.)
- [x] E2E workflow (generate -> guard -> audit)
- [x] Benchmark and Privacy sampling

### Phase 3: "The Professional"

- [x] CI/CD optimization and Release workflow
- [x] Advanced Generator (CTGAN, optional)
- [x] ML Utility evaluation (TSTR: AUC/F1)
- [x] PyPI metadata/CHANGELOG/README maintenance

### Future Ideas

- Rule Learning Engine (Reinforcement Learning based)
- Auto-Tuning Mode (Autonomous quality improvement)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Write tests before implementing (TDD).
4. Commit your changes (`git commit -m 'Add amazing feature'`).
5. Push to the branch (`git push origin feature/amazing-feature`).
6. Create a Pull Request.

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions regarding the project, please create an Issue.

## Acknowledgments

- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV)
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- Kaggle Credit Card Fraud Detection Dataset
