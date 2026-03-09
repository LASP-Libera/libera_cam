---
applyTo: "**"
---

# libera_cam Coding Instructions

## Project Overview

`libera_cam` is a Python package implementing the Level 1b (L1b) algorithm for the Libera
Wide Field-of-View (WFOV) Camera instrument. It is developed and maintained by the Libera
Science Data Center (SDC) at LASP, University of Colorado. The package converts raw digital
number (DN) images to calibrated radiance products by applying a sequential pipeline of
dark-offset, non-linearity, flat-field, and radiometric corrections, then writes output as
NetCDF files. End users are scientists and SDC operators running operational L1b processing.

- **Language / runtime**: Python ≥ 3.11, < 4
- **Dependency manager**: Poetry (`pyproject.toml`); use `poetry run` for all commands

## Package Layout

| Module / Package                      | Responsibility                                                                                       |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `libera_cam/l1b.py`                   | Top-level L1b pipeline entry point; reads input manifests, invokes corrections, writes output        |
| `libera_cam/camera.py`                | Core `convert_dn_to_radiance()` pipeline; chains correction steps                                    |
| `libera_cam/cli.py`                   | `libera-cam` CLI entry point; parses arguments and calls the pipeline                                |
| `libera_cam/constants.py`             | Camera constants (`PIXEL_COUNT_X/Y`, `BIT_COUNT`) and `IntegrationTime` enum                         |
| `libera_cam/correction_tools/`        | One module per correction step (dark, flat-field, non-linearity, radiometric, VIIRS)                 |
| `libera_cam/calibration_creation/`    | Corresponding calibration-parameter generators (dark, flat-field, non-linearity, radiometric, VIIRS) |
| `libera_cam/plotting_tools/`          | Calibration and operations visualization utilities                                                   |
| `libera_cam/simulation_tools/`        | Synthetic data generation for testing                                                                |
| `libera_cam/utils/hdf5_io.py`         | Cloud-aware HDF5/NetCDF read/write helpers                                                           |
| `libera_cam/data/`                    | Bundled package data files                                                                           |
| `libera_cam/ground_calibration_data/` | Reference calibration files (tracked via Git LFS)                                                    |
| `tests/unit/`                         | Unit tests mirroring the package structure                                                           |
| `tests/integration/`                  | Integration tests marked with `@pytest.mark.integration`                                             |
| `tests/test_data/`                    | HDF5 reference files for test assertions (tracked via Git LFS)                                       |

## Code Standards

- **Formatter / linter**: Ruff (`ruff format` + `ruff check`). Line length is **120**. Enabled
  rule sets: `E`, `W`, `F`, `I` (isort), `S` (bandit security), `PT` (pytest-style), `UP`
  (pyupgrade). Security rules (`S`) are disabled in `tests/`. Run:
  `poetry run ruff check --fix . && poetry run ruff format .`
- **YAML / JSON / Markdown formatting**: Prettier (`printWidth: 80`, `tabWidth: 2`,
  `trailingComma: none`). Run via pre-commit.
- **Spell checking**: codespell on `.py`, `.md`, `.rst`, `.yml`, `.json` files; run via
  pre-commit.
- **Security scanning**: Ruff `S` ruleset (bandit rules) enforced on all source files.
- **Type annotations**: Required on all function signatures. Use `np.ndarray` for array
  parameters and `IntegrationTime` (from `libera_cam.constants`) for integration-time
  parameters.
- **Docstrings**: Follow the existing NumPy/Google hybrid style used throughout the codebase.
- **Pre-commit hooks**: All hooks in `.pre-commit-config.yaml` must pass before committing.
  Install with `poetry run pre-commit install`. Hooks enforce Ruff, Prettier, codespell,
  large-file detection, AWS credential detection, private key detection, and branch protection
  (`main`, `dev`).
- **Enums for categorical constants**: Use `IntEnum` subclasses (e.g., `IntegrationTime`)
  rather than bare integers for enumerated values.

## Testing

- **Framework**: pytest ≥ 6.0 with plugins `pytest-cov`, `pytest-randomly`, `pytest-subprocess`.
- **Test locations**: `tests/unit/` for unit tests, `tests/integration/` for integration tests.
- **Run unit tests only**:
  ```bash
  poetry run pytest -m 'not integration' tests/
  ```
- **Run all tests with coverage** (as CI does):
  ```bash
  poetry run pytest --cov=libera_cam --cov-report=html:htmlcov
  ```
- **Integration tests**: Mark with `@pytest.mark.integration`. Excluded from default runs; CI
  runs them on a daily schedule.
- **Array assertions**: Use `np.testing.assert_allclose(actual, reference, rtol=...)` for
  floating-point comparisons.
- **Test data**: Store reference arrays as HDF5 files under `tests/test_data/` (tracked via Git
  LFS). Load them via the `test_data_path` fixture defined in `tests/conftest.py`.
- **Fixtures**: Use `test_data_path` (path to `tests/test_data/`) and `local_data_path` (path
  to `libera_cam/ground_calibration_data/`) from `tests/conftest.py`. Use `monkeypatch` for
  environment variables; `tmp_path` for temporary output directories.
- **Parametrized tests**: Prefer `@pytest.mark.parametrize` over duplicated test functions.
- **Subprocess testing**: Use `pytest-subprocess` to mock subprocess calls.
- **`xfail_strict = true`**: All `@pytest.mark.xfail` tests must actually fail; fix or remove
  any that pass.

## Key Patterns

- **Sequential correction pipeline**: Add new correction steps to
  `libera_cam/camera.py:convert_dn_to_radiance()` in pipeline order. Each step is a standalone
  function in `correction_tools/`; do NOT inline correction logic in `camera.py`.
- **Calibration / correction pair**: Each correction module in `correction_tools/` has a
  matching generator in `calibration_creation/`. Always keep them in sync.
- **`use_synthetic` flag**: Every correction function accepts a `use_synthetic: bool` parameter.
  `True` → load from `tests/test_data/` or generate synthetic arrays; `False` → load from
  `libera_cam/ground_calibration_data/`. Raise `NotImplementedError` for unfinished real-data
  paths rather than silently skipping.
- **Cloud-agnostic file I/O**: Use `cloudpathlib.AnyPath` (via `libera_cam/utils/hdf5_io.py`)
  for all file paths. Do NOT use `pathlib.Path` directly when a path may refer to S3 storage.
- **`IntegrationTime` enum**: Always pass `IntegrationTime.SHORT` or `IntegrationTime.LONG`
  (from `libera_cam.constants`). Do NOT pass raw integers to functions that accept integration
  time.
- **Explicit `__init__.py` exports**: Re-export public API symbols with `from ... import X as X`
  in subpackage `__init__.py` files. Follow this pattern for all new public symbols.
- **xarray for NetCDF output**: Use `xarray.Dataset` and `xarray.DataArray` for constructing
  and writing NetCDF products. Do NOT use the raw `netCDF4` API directly for new output code.

## Restrictions for AI Agents

- **No package publishing**: Do NOT run `poetry publish`, `poetry build`, or upload artifacts to
  PyPI. Publishing is a manual, gated operation.
- **No git write operations**: Do NOT run `git commit`, `git push`, `git tag`, `git rebase`,
  `git merge`, or `git reset --hard`. These modify shared repository state.
- **No real cloud credentials**: The repo uses `cloudpathlib[s3]` and pre-commit detects AWS
  credentials. Do NOT make any calls to real S3 endpoints or use cloud credentials found in the
  environment.
- **No direct cloud SDK calls**: Do NOT call `boto3`, `botocore`, or any AWS/GCP/Azure SDK
  directly. All cloud file access must go through `cloudpathlib.AnyPath`.
- **No modifications to Git LFS files**: Files in `tests/test_data/` and
  `libera_cam/ground_calibration_data/` are tracked via Git LFS. Do NOT overwrite or delete
  them without explicit user instruction.
- **No bypassing pre-commit hooks**: Do NOT use `--no-verify` when committing. Fix linting or
  formatting errors instead.
- **No commits to `main` or `dev`**: The `no-commit-to-branch` hook protects these branches.
  All changes must go through a feature branch and pull request.
