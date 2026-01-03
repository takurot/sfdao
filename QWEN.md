# SFDAO - Synthetic Finance Data Auditor & Optimizer

## Project Overview

SFDAO is a comprehensive synthetic data platform for the financial industry, designed to generate, constrain, and audit synthetic financial data. The project is structured in three phases:

- **Phase 1: "The Auditor" (MVP)** - Core audit functionality for evaluating synthetic data quality
- **Phase 2: "The Generator & Logic"** - Data generation with constraint application and scenario injection
- **Phase 3: "The Professional"** - Advanced features including ML utility evaluation and optimization

Key features include:
- Statistical quality evaluation (KS test, Jensen-Shannon Divergence)
- Financial-specific evaluation (Fat Tail detection, Volatility Clustering)
- Privacy evaluation (re-identification risk, Distance to Closest Record)
- Automatic type detection (numeric, categorical, datetime, PII)
- Generation workflows with `generate`/`run` commands
- Constraint and scenario application
- ML Utility evaluation (TSTR with AUC/F1)
- HTML/PDF report generation

## Building and Running

### Prerequisites
- Python 3.10+
- Poetry (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/takurot/sfdao.git
cd sfdao

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies based on pyproject.toml/poetry.lock
poetry install

# Activate virtual environment (optional)
poetry shell

# For macOS: Install WeasyPrint dependencies for PDF generation
brew install cairo pango gdk-pixbuf libffi
```

### Running the Application
```bash
# Basic audit evaluation
sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.html

# Generate synthetic data
poetry run sfdao generate --config config.yaml --real data/real.csv --output synthetic.csv

# Run full pipeline (generate → guard → audit)
poetry run sfdao run --config config.yaml --real data/real.csv --out-dir output/
```

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
# Format check
black --check .

# Apply formatting
black .

# Lint check
flake8 .

# Type checking
mypy sfdao

# Security check
bandit -r sfdao
```

## Development Conventions

### TDD (Test-Driven Development)
The project follows TDD methodology with the cycle:
1. **Red**: Write a failing test
2. **Green**: Write minimal code to pass the test
3. **Refactor**: Optimize and clean up the code

### Branch Strategy
- Always create dedicated branches for PR work
- Naming convention: `feature/pr-XX-short-description`
- Never work directly on the main branch

### Code Quality Standards
- Test coverage target: 90%+
- Type checking: mypy in strict mode
- Formatting: Black
- Linting: Flake8
- Security: Bandit

### Project Structure
```
sfdao/
├── sfdao/                  # Main package
│   ├── ingestion/          # Data ingestion and type detection
│   ├── config/             # Configuration schemas/loaders
│   ├── generator/          # Synthetic data generation
│   ├── guard/              # Rule-based constraint checking
│   ├── scenario/           # Scenario injection
│   ├── evaluator/          # Evaluation metric calculation
│   ├── reporter/           # Report generation
│   └── cli/                # CLI interface
├── tests/                  # Test code
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-End tests
├── docs/                   # Documentation
└── prompt/                 # Specification documents
```

## Key Files and Directories

- `pyproject.toml` - Project configuration and dependencies
- `README.md` - Project overview and quick start guide
- `prompt/PLAN.md` - Detailed implementation plan
- `example/` - Example usage and test cases
- `tests/` - Comprehensive test suite
- `docs/` - Documentation files