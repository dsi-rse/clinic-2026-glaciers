# Python Development Guidelines

## Environment management
- Manage the Python environment with uv.
- Manage additional dependencies with Docker.

## Repo structure
- Core Python code lives in src/{package-name}/.

### Runnable Python scripts
- Runnable scripts live in scripts/.
- Runnable scripts should use argparse.
- Core functionality that is needed for runnable scripts should be imported from {package-name}.
- All scripts should be documented in README.md.
- Scripts that users and developers may have to run regularly should be runnable via a `make` command, specified in the Makefile.

### Python notebooks
- Python notebooks live in notebooks/
- Core functionality that is needed for notebooks should be imported from {package-name}

### Data
- Data should be read from and written to the directory specified in settings.DATA_DIR.
  - DATA_DIR should have a documented and well-organized structure that clearly distinguishes different types of files.
  - Runnable scripts can take data input and output paths as arguments, but should have defaults that are standard locations in DATA_DIR.

## Code quality & conventions
- Code should follow the standards described here: https://clinic.ds.uchicago.edu/coding-standards/coding-standards.html.
- All code must pass the [`ruff`](https://docs.astral.sh/ruff/) rules as defined in pyproject.toml.
  - `ruff` should be run before each commit via [`pre-commit`](https://pre-commit.com/). If it fails, the commit will be blocked and the user will be shown what needs to be changed.
  - To check for errors locally, run `uv run pre-commit install` once. Then check for errors by running: `uv run pre-commit run --all-files`.

