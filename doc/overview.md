# libera_cam Overview

Operator and developer guide for running the WFOV camera L1B pipeline.

**Related documentation**

- [changelog.md](changelog.md) — shipped changes by version
- [roadmap.md](roadmap.md) — forward-looking L1B plans
- [wfov_fsw_header_reference.md](wfov_fsw_header_reference.md) — FSW metadata and L1A time coordinates

---

## L1B processing

The L1B pipeline is invoked from an input manifest (see `libera_cam/l1b.py` and the
`libera-cam` CLI). Set **`PROCESSING_PATH`** to the output directory before running.

```bash
export PROCESSING_PATH=/path/to/output
libera-cam /path/to/input_manifest.json
```

Manifest **`configuration`** keys (optional):

| Key         | Default | Effect                                                                 |
| ----------- | ------- | ---------------------------------------------------------------------- |
| `use_geo`   | `true`  | When `false`, skip SPICE kernels and write placeholder geolocation     |
| `jpss_only` | absent  | When `true`, use JPSS-only SPICE kernels and `LIBERA_BASE` geolocation |

`use_geo: false` and `jpss_only: true` cannot both be set.

---

## Dask parallelization

L1B uses Dask for lazy L1A decompression, radiometry, and geolocation. Tune execution with
environment variables before starting the pipeline:

| Variable                | Default       | Description                                                                                                                                                                                                       |
| ----------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DASK_SCHEDULER`        | `synchronous` | Dask scheduler. **`synchronous`** (single-process, default) or **`distributed`** (local cluster with dashboard). **`threads`** and **`processes`** are not supported — CSPICE is not thread-safe within a worker. |
| `DASK_NUM_WORKERS`      | `1`           | Number of Dask workers when using `distributed`.                                                                                                                                                                  |
| `DASK_MEMORY_LIMIT`     | `8GB`         | Per-worker memory limit for `distributed` (e.g. `4GB`, `16GB`).                                                                                                                                                   |
| `LIBERA_CAM_CHUNK_SIZE` | `50`          | Number of L1A images per Dask batch during JPEG-LS decompression in `read_l1a_cam_data`. Lower values reduce peak memory; higher values reduce scheduler overhead.                                                |

### Examples

Single-process run (default, suitable for tests and small products):

```bash
export PROCESSING_PATH=/path/to/output
export DASK_SCHEDULER=synchronous
libera-cam /path/to/input_manifest.json
```

Multi-worker local cluster (larger products; watch memory per worker):

```bash
export PROCESSING_PATH=/path/to/output
export DASK_SCHEDULER=distributed
export DASK_NUM_WORKERS=4
export DASK_MEMORY_LIMIT=8GB
export LIBERA_CAM_CHUNK_SIZE=50
libera-cam /path/to/input_manifest.json
```

When `DASK_SCHEDULER=distributed`, the pipeline logs a Bokeh dashboard URL for task profiling.

### Profiling in tests

Integration test `tests/integration/test_l1b_profiling.py` runs a full L1B smoke test with
Dask `Profiler` output printed to stdout (visible in CI logs). Optional: set
`L1B_PROFILE_VISUALIZE=1` to write HTML profile artifacts under the test temp directory.

To build a larger L1A fixture for local profiling, use the `make_extended_l1a` pytest fixture
(see `tests/helpers/l1a_scaling.py` and `tests/conftest.py`).
