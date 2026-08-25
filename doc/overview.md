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

## L1A inputs

L1B requires image-centric WFOV L1A products from libera-utils ≥ 5.10.5. Complete SOP→EOP images
are stitched onto a `CAMERA_TIME` dimension, JPEG-LS payloads are stored in
`WFOV_COMPRESSED_IMAGE` with per-image valid byte lengths in `WFOV_COMPRESSED_IMAGE_LENGTH`, and
FSW/FPGA header metadata is decoded into `WFOV_FSW_HEADER_*`, `WFOV_IMAGE_HEADER_*`,
`WFOV_IMAGE_FOOTER_*`, and `WFOV_FPGA_STATUS_*` variables. A packet-only L1A product raises a
`ValueError` naming the missing fields.

Images whose `WFOV_HEADER_PARSE_VALID` is false are dropped with a logged warning — they carry a
NaT `CAMERA_TIME` and cannot be geolocated. L1B raises if no image survives.

L1A header fields reach the L1B product as:

| L1B variable                    | L1A source                                   | Notes                           |
| ------------------------------- | -------------------------------------------- | ------------------------------- |
| `Radiometer_Observation_ID`     | `WFOV_FSW_HEADER_RAD_OBS_ID`                 |                                 |
| `Camera_Observation_ID`         | `WFOV_FSW_HEADER_CAM_OBS_ID`                 |                                 |
| `Image_Mode`                    | `WFOV_FSW_HEADER_IMG_MODE`                   | 0=DUAL, 1=VIDEO, 2=IMGA, 3=IMGB |
| `Camera_Packet_Index`           | `CAMERA_PACKET_INDEX`                        | Originating L1A packet index    |
| `Actual_Exposure_Time_1` / `_2` | `WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1` / `_2` | Converted to milliseconds       |
| `Exposure_Delta`                | `WFOV_IMAGE_HEADER_DELTA`                    | Converted to milliseconds       |

Exposure register conversions are documented in
[wfov_fsw_header_reference.md](wfov_fsw_header_reference.md#exposure-timing).

`CAMERA_TIME` is not unique. In VIDEO mode (`Image_Mode == 1`) one camera trigger produces two
images sharing a single FSW timestamp; `Camera_Packet_Index` distinguishes them. See
[roadmap.md](roadmap.md) for the planned resolution.

---

## Dask parallelization

L1B uses Dask for lazy L1A decompression, radiometry, and geolocation. Tune execution with
environment variables before starting the pipeline:

| Variable                | Default       | Description                                                                                                                                                                                                       |
| ----------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DASK_SCHEDULER`        | `synchronous` | Dask scheduler. **`synchronous`** (single-process, default) or **`distributed`** (local cluster with dashboard). **`threads`** and **`processes`** are not supported — CSPICE is not thread-safe within a worker. |
| `DASK_NUM_WORKERS`      | `1`           | Number of Dask workers when using `distributed`.                                                                                                                                                                  |
| `DASK_MEMORY_LIMIT`     | `8GB`         | Per-worker memory limit for `distributed` (e.g. `4GB`, `8GB`, `16GB`).                                                                                                                                            |
| `LIBERA_CAM_CHUNK_SIZE` | `50`          | Number of L1A images per Dask batch during JPEG-LS decompression in `read_l1a_cam_data`. Lower values reduce peak memory; higher values reduce scheduler overhead.                                                |

### Operator Tuning Guide

When sizing AWS Batch containers and configuring Dask execution, keep the following relationships in mind:

1. **Production Scheduling**: Shipped defaults (`synchronous`, `DASK_NUM_WORKERS=1`) run single-threaded for safe local testing. Production Batch job definitions should explicitly configure `DASK_SCHEDULER=distributed` along with `DASK_NUM_WORKERS` matched to the allocated container vCPUs.
2. **Parallelism is Capped by Chunk Count**: Geolocation blocks align 1:1 with decompression chunks (`add_geolocation_to_dataset` reuses `ds.image_data.chunks[0]`), and inside each worker block, frames are evaluated serially for SPICE thread-safety. Therefore, the maximum number of useful workers is:
   $$\text{max\_useful\_workers} = \left\lceil \frac{N_{\text{images}}}{\text{LIBERA\_CAM\_CHUNK\_SIZE}} \right\rceil$$
   For example, a 500-image product with `LIBERA_CAM_CHUNK_SIZE=50` creates 10 chunks and can utilize at most 10 workers. To scale across wider containers (e.g. 32 vCPUs), `LIBERA_CAM_CHUNK_SIZE` should be decreased accordingly (e.g. 15–20).
3. **Worker Memory Sizing**: Each active worker processes one chunk at a time. For `LIBERA_CAM_CHUNK_SIZE=50`, memory per chunk comprises ~800 MB for `image_data` plus ~2.5 GB for `float32` geolocation (lat, lon, alt), totaling ~3.3 GB live task memory. Setting `DASK_MEMORY_LIMIT=8GB` provides safe operating headroom for SPICE calculations and decompression buffers. Sizing rule of thumb:
   $$\text{Total System Memory} \ge (\text{DASK\_NUM\_WORKERS} \times \text{DASK\_MEMORY\_LIMIT}) + \text{Client Overhead}$$
4. **Client Write Funneling**: When `write_data_product` writes the final NetCDF file via `smart_open` / `h5netcdf`, computed chunks are funneled through the client process. Ensure the client host process has adequate memory headroom.

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
