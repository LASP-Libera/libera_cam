# libera_cam Overview

Operator and developer guide for running the WFOV camera L1B pipeline.

**Related documentation**

- [CHANGELOG.md](../CHANGELOG.md) — shipped changes by version
- [performance.md](performance.md) — measured sizing, throughput, memory and encoding behind the defaults
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

With `use_geo` true, the run raises `ValueError` if the spacecraft SPK/CK, or outside
`jpss_only` the AZROT-CK, cover none of the granule's frames. Frames a partly covered
granule misses are written as NaN, and a warning gives their count for each variable.
`jpss_only` writes the three `Azimuth_Motor_Position_SPICE` variables as 0 degrees. The
global attribute `Geolocation_Mode` names the mode: `production`, `jpss_only` or
`use_geo_false`.

---

## L1A inputs

L1B requires image-centric WFOV L1A products from libera-utils ≥ 5.10.5. Complete SOP→EOP images
are stitched onto a `CAMERA_TIME` dimension, JPEG-LS payloads are stored in
`WFOV_COMPRESSED_IMAGE` with per-image valid byte lengths in `WFOV_COMPRESSED_IMAGE_LENGTH`, and
FSW/FPGA header metadata is decoded into `WFOV_FSW_HEADER_*`, `WFOV_IMAGE_HEADER_*`,
`WFOV_IMAGE_FOOTER_*`, and `WFOV_FPGA_STATUS_*` variables. A packet-only L1A product raises a
`ValueError` naming the missing fields.

L1A header fields reach the L1B product as:

| L1B variable                          | L1A source                                   | Notes                           |
| ------------------------------------- | -------------------------------------------- | ------------------------------- |
| `Azimuth_Motor_Position_FSW_Metadata` | `WFOV_FSW_HEADER_AZIMUTH_ANGLE`              | Degrees, signed about zero      |
| `Radiometer_Observation_ID`           | `WFOV_FSW_HEADER_RAD_OBS_ID`                 |                                 |
| `Camera_Observation_ID`               | `WFOV_FSW_HEADER_CAM_OBS_ID`                 |                                 |
| `Image_Mode`                          | `WFOV_FSW_HEADER_IMG_MODE`                   | 0=DUAL, 1=VIDEO, 2=IMGA, 3=IMGB |
| `Camera_Packet_Index`                 | `CAMERA_PACKET_INDEX`                        | Originating L1A packet index    |
| `Actual_Exposure_Time_1` / `_2`       | `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1` / `_2` | Converted to milliseconds       |
| `Exposure_Delta`                      | `WFOV_IMAGE_HEADER_DELTA`                    | Converted to milliseconds       |

Exposure register conversions are documented in
[wfov_fsw_header_reference.md](wfov_fsw_header_reference.md#exposure-timing).

`CAMERA_TIME` is not unique. In VIDEO mode (`Image_Mode == 1`) one camera trigger produces two
images sharing a single FSW timestamp; `Camera_Packet_Index` distinguishes them. See
[roadmap.md](roadmap.md) for the planned resolution.

---

## Dask parallelization

L1B uses Dask for lazy L1A decompression, radiometry, and geolocation. Tune execution with
environment variables before starting the pipeline:

| Variable                    | Default       | Description                                                                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `XARRAY_NETCDF_ENGINE`      | `h5netcdf`    | NetCDF engine used by `libera_utils` to write the product. Both engines write to S3. `distributed` writes through worker compression, which requires `h5netcdf`: libera-utils raises `ValueError` under `netcdf4`. `h5netcdf` is also faster and produces smaller granules (see [performance.md](performance.md#tuning-relationships)).                                              |
| `DASK_SCHEDULER`            | `synchronous` | Dask scheduler. **`synchronous`** (single-process, default) or **`distributed`** (local cluster with dashboard). **`threads`** and **`processes`** are not supported — CSPICE is not thread-safe within a worker.                                                                                                                                                                    |
| `DASK_NUM_WORKERS`          | `7`           | Number of Dask workers when using `distributed`. Each worker computes and compresses its own blocks, so the write speeds up with workers until they fill the task's vCPUs; seven suits the 8 vCPU / 60 GB task (see [performance.md](performance.md#aws-sizing)).                                                                                                                    |
| `DASK_MEMORY_LIMIT`         | `8GB`         | Per-worker memory limit for `distributed`. It must cover one worker's worst moment, which the decompression chunk sets: at the default chunk of 20 a worker peaked at 4.3-7.2 GB, measured with HDF5 compressing in the writing process. With the workers compressing, the full July 12 day peaked at 15.8 GB for the whole task at the defaults. Dask pauses a worker at 80% of it. |
| `LIBERA_CAM_CHUNK_SIZE`     | `20`          | Number of L1A images per Dask batch during JPEG-LS decompression in `read_l1a_cam_data`. Lower values reduce peak memory; higher values reduce scheduler overhead.                                                                                                                                                                                                                   |
| `LIBERA_CAM_GEO_CHUNK_SIZE` | `5`           | Number of frames per per-pixel geometry task in `add_geolocation_to_dataset`, independent of the decompression chunk. Each task furnishes the kernels once and geolocates its frames one at a time.                                                                                                                                                                                  |

### Production settings

Production runs `DASK_SCHEDULER=distributed` with the default `h5netcdf` engine on an
8 vCPU / 60 GB task: seven workers at `8GB`, `LIBERA_CAM_CHUNK_SIZE=20` and
`LIBERA_CAM_GEO_CHUNK_SIZE=5`. The pipeline's `LocalCluster` runs one thread per worker,
because CSPICE is not thread-safe. Size the container against the whole process tree,
not the client: at these settings the full July 12 day (8,205 frames) peaked at 15.8 GB.
Every worker reads the L1A file by path, so a multi-node cluster needs shared filesystem
access to it. The measurements and tuning relationships behind these settings are in
[performance.md](performance.md).

### Examples

Single-process run (default, suitable for tests and small products):

```bash
export PROCESSING_PATH=/path/to/output
export DASK_SCHEDULER=synchronous
libera-cam /path/to/input_manifest.json
```

Multi-worker local cluster at the defaults, which are sized for an 8 vCPU / 60 GB task. Seven
workers at `8GB` may use up to 56 GB between them; the full July 12 day peaked at 15.8 GB with
the workers compressing. On a smaller machine, use one worker per core. On a 16 GB machine,
two workers at `6GB` with `LIBERA_CAM_CHUNK_SIZE=10` ran cleanly with HDF5 compressing in the
writing process (see [performance.md](performance.md#aws-sizing)):

```bash
export PROCESSING_PATH=/path/to/output
export DASK_SCHEDULER=distributed
export DASK_NUM_WORKERS=7
export DASK_MEMORY_LIMIT=8GB
export LIBERA_CAM_CHUNK_SIZE=20
export LIBERA_CAM_GEO_CHUNK_SIZE=5
libera-cam /path/to/input_manifest.json
```

The product version in the output filename and `algorithm_version` comes from the installed
package metadata, not from `pyproject.toml`, so reinstall (`pip install -e .`) after switching
branches; a stale editable install labels the product with whatever version it last recorded.

When `DASK_SCHEDULER=distributed`, the pipeline logs a Bokeh dashboard URL for task profiling.
Note that `DASK_SCHEDULER`, `DASK_NUM_WORKERS` and `DASK_MEMORY_LIMIT` are also read by Dask's
own configuration, so `DASK_SCHEDULER=distributed` set in the environment makes any Dask
compute outside the pipeline's client context fail with `no Client active`.

### Profiling in tests

Integration test `tests/integration/test_l1b_profiling.py` runs a full L1B smoke test with
Dask `Profiler` output printed to stdout (visible in CI logs). Optional: set
`L1B_PROFILE_VISUALIZE=1` to write HTML profile artifacts under the test temp directory.

To build a larger L1A fixture for local profiling, use the `make_extended_l1a` pytest fixture
(see `tests/helpers/l1a_scaling.py` and `tests/conftest.py`).
