# Contributing to sfdao

Thank you for your interest in contributing to sfdao! We welcome contributions from the community to help improve this project.

## Development Process

1.  **Fork and Clone**: Fork the repository and clone it locally.
2.  **Create a Branch**: Create a new branch for your feature or fix.
    ```bash
    git checkout -b feature/my-feature
    ```
3.  **Install Dependencies**: Use Poetry to install dependencies.
    ```bash
    poetry install
    ```
4.  **Implement Changes**: Make your changes, following `TDD` guidelines.
5.  **Test**: Run tests to ensure your changes work as expected.
    ```bash
    poetry run pytest
    ```
6.  **Lint**: Run code quality checks.
    ```bash
    poetry run black .
    poetry run flake8 .
    poetry run mypy sfdao
    poetry run bandit -r sfdao
    ```
7.  **Submit PR**: Push your changes and create a Pull Request.

## CI/CD Checks

All Pull Requests must pass the following checks:
-   **Tests**: All unit and integration tests must pass on Python 3.10, 3.11, and 3.12.
-   **Linting**: Code must be formatted with Black and pass Flake8 checks.
-   **Type Checking**: Mypy strict mode must pass with no errors.
-   **Security**: Bandit security checks must pass.

## Release Process

Releases are automated via GitHub Actions:
1.  Create a new tag starting with `v` (e.g., `v0.1.0`).
2.  Push the tag to the repository.
3.   The `release` workflow will automatically:
    -   Build the package.
    -   Publish to PyPI (requires Trusted Publishing configuration).
    -   Create a GitHub Release with auto-generated notes.
