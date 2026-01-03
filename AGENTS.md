# Repository Guidelines

## Project Structure & Module Organization
- `sfdao/` contains the core Python package, organized by domain: `ingestion/`, `generator/`, `guard/`, `evaluator/`, `reporter/`, and `cli/`.
- `tests/` is split into `unit/`, `integration/`, and `e2e/` suites.
- `docs/` holds user-facing documentation, while `prompt/` contains product specs and plans.
- `example/` provides sample configs, data, and output to verify end-to-end flows.

## Build, Test, and Development Commands
- Install dependencies with Poetry: `poetry install`.
- Run the CLI locally: `poetry run sfdao audit --real data/real.csv --synthetic data/synthetic.csv --output report.html`.
- Execute the pipeline workflow: `poetry run sfdao run --config example/config/phase2.yaml --outdir example/output`.
- macOS PDF support requires `brew install cairo pango gdk-pixbuf libffi` for WeasyPrint.

## Coding Style & Naming Conventions
- Python 3.10+ only; use 4-space indentation and standard type hints.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, and `SCREAMING_SNAKE_CASE` for constants.
- Formatting and checks: `black .`, `flake8 .`, `mypy sfdao`, and `bandit -r sfdao`.

## Testing Guidelines
- Test with `pytest` and prefer TDD (red → green → refactor).
- Run all tests: `pytest`.
- Coverage report: `pytest --cov=sfdao --cov-report=html`.
- Target a file: `pytest tests/unit/ingestion/test_loader.py`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits seen in history: `type: subject` or `type(scope): subject` (e.g., `docs: update README`, `chore(release): prepare for v0.1.0`).
- PRs should include a clear summary, linked issues (if any), and test results; add screenshots when report output changes.

## Configuration & Security Notes
- Example configs live under `example/config/`; keep new schemas in `sfdao/config/`.
- Handle real data carefully; store local datasets outside version control and use `example/data/` for samples.
