# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-19

### Added
- **Guard**: `fill_to_target` option for `exclude` mode — resamples until `n_samples` rows satisfy guard constraints, ensuring consistent output size (#38).
- **CLI**: `--ml-utility` and `--ml-target` flags added to `sfdao run` command for combined generate→guard→audit workflows (#36).
- **Generator**: `params` from `GeneratorSettings` are now passed through to generator constructors, enabling full configuration via YAML (#37).

### Changed
- **Config**: `ScenarioSettings` simplified — `name` and `transformations` are now direct fields instead of being nested under `params`, reducing config hierarchy depth (#39).

## [0.1.0] - 2025-12-31

### Added
- **Core**:
  - Initial release of `sfdao` (Synthetic Finance Data Auditor & Optimizer).
  - Data ingestion with automatic schema extraction and type detection (`sfdao.ingestion`).
  - Financial domain masking and role inference.
- **Generator**:
  - Baseline Generator (Statistical sampling) for rapid data generation.
  - Advanced Generator (CTGAN integration) via optional `[deep]` extra.
  - Scenario Engine for injecting specific patterns (outliers, trends).
  - Guard Engine for enforcing business rules and constraints.
- **Evaluator**:
  - Statistical quality metrics (KS Test, JS Divergence).
  - Financial stylized facts verification (Fat tails, Volatility clustering).
  - Privacy risk assessment (Distance to Closest Record, Re-identification risk).
  - Machine Learning Utility evaluation (TSTR - Train on Synthetic, Test on Real).
- **CLI**:
  - `sfdao audit`: Comprehensive evaluation of synthetic data against real data.
  - `sfdao generate`: Generate synthetic data using Baseline or CTGAN models.
  - `sfdao run`: Combined workflow (Generate -> Guard -> Audit).
- **Reporting**:
  - HTML and PDF evaluation reports with visualizations.
  - Composite scoring system.
- **Documentation**:
  - Comprehensive documentation in `docs/`.
  - Example project in `example/` for quick start.
