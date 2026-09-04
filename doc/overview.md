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

| Variable                    | Default       | Description                                                                                                                                                                                                                     |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `XARRAY_NETCDF_ENGINE`      | `h5netcdf`    | NetCDF engine used by `libera_utils` to write the product. AWS runs set **`netcdf4`**, which is what makes `distributed` work at all (see the tuning guide). `h5netcdf` is the only engine that can write to S3 paths directly. |
| `DASK_SCHEDULER`            | `synchronous` | Dask scheduler. **`synchronous`** (single-process, default) or **`distributed`** (local cluster with dashboard). **`threads`** and **`processes`** are not supported — CSPICE is not thread-safe within a worker.               |
| `DASK_NUM_WORKERS`          | `1`           | Number of Dask workers when using `distributed`.                                                                                                                                                                                |
| `DASK_MEMORY_LIMIT`         | `8GB`         | Per-worker memory limit for `distributed` (e.g. `4GB`, `8GB`, `16GB`).                                                                                                                                                          |
| `LIBERA_CAM_CHUNK_SIZE`     | `50`          | Number of L1A images per Dask batch during JPEG-LS decompression in `read_l1a_cam_data`. Lower values reduce peak memory; higher values reduce scheduler overhead.                                                              |
| `LIBERA_CAM_GEO_CHUNK_SIZE` | `10`          | Number of frames per per-pixel geometry task in `add_geolocation_to_dataset`, independent of the decompression chunk. Each task furnishes the kernels once and geolocates its frames one at a time.                             |

### Operator Tuning Guide

When sizing AWS Batch containers and configuring Dask execution, keep the following relationships in mind:

1. **Production Scheduling**: `DASK_SCHEDULER=distributed` requires `XARRAY_NETCDF_ENGINE=netcdf4`, which is the pairing AWS runs use and the one measured below. With the default `h5netcdf` engine, `distributed` fails in `write_data_product` with `TypeError: cannot pickle '_io.BufferedRandom' object`: that branch of `libera_utils.io.netcdf.write_libera_data_product` hands `to_netcdf` an open file object, and the distributed scheduler must serialize the store targets to its workers. The `netcdf4` branch passes a path instead and has no such object. The failure is independent of geolocation — it reproduces with `use_geo: false` — and the engine switch is a workaround rather than a diagnosis: the reason the write path holds an unpicklable handle has not been addressed, and `netcdf4` cannot write to S3 paths, so a run whose `PROCESSING_PATH` is an S3 URI still needs `h5netcdf` and therefore `synchronous`. Roadmap item 7 carries the comparison. Measured with dask/distributed 2026.8.0, xarray 2026.7.0, h5netcdf 1.8.1.
2. **Parallelism is Capped by Task Count** (applies once distributed writing works): Decompression runs in `ceil(N_images / LIBERA_CAM_CHUNK_SIZE)` tasks and per-pixel geometry in `ceil(N_images / LIBERA_CAM_GEO_CHUNK_SIZE)` tasks; inside a geometry task, frames are evaluated serially for SPICE thread-safety. The maximum number of useful workers for each stage is its task count:
   $$\text{max\_useful\_workers} = \left\lceil \frac{N_{\text{images}}}{\text{chunk size}} \right\rceil$$
   For example, a 500-image product with `LIBERA_CAM_GEO_CHUNK_SIZE=10` creates 50 geometry tasks and can utilize at most 50 workers on that stage. Smaller chunks add per-task kernel furnishing and scheduler overhead.
3. **Task Memory Sizing**: Each active worker, or the single process under `synchronous`, holds one task at a time. A geometry task holds its output block, 143 MB per frame of eight `float32` fields and the `uint16` flags, on top of the kernels, pixel vectors and the curryer transients for the frame in progress. Measured by calling `calculate_chunk_geometry` directly in a fresh process, one task of `N` frames peaks at 2.21, 2.51, 2.99, 3.74 and 5.19 GB for `N` of 1, 2, 5, 10 and 20, a least-squares fit of

   $$\text{task peak RSS} \approx 2.2\,\text{GB} + 0.15\,\text{GB} \times N_{\text{frames in the task}}$$

   Per-frame cost falls from 2.08 s at one frame to 1.00 s asymptotically, so the ~1.1 s of per-task kernel furnishing is 6% overhead at the default chunk of 10 and 17% at 5. A decompression task at `LIBERA_CAM_CHUNK_SIZE=50` holds ~800 MB of `image_data` and masks. This bounds one task, not one run: whole-run peak memory is set by the write path instead, and is the subject of "Granule length is capped by client memory" below, which gives the full container sizing formula.

4. **Client Write Funneling**: Computed blocks are funneled through the client process when the product is written, and are not released as they are written. This sets the granule length a container can handle, on both write paths; see "Granule length is capped by client memory" below.

### Data volume and throughput

Measured on the `DITL_3min` granule (three 2048x2048 frames, 21.7% of pixels off the
ellipsoid), single core, local SSD, with the default `h5netcdf` engine:

| Quantity                            | Per frame | 12 h granule at 5 s (8640 frames)   |
| ----------------------------------- | --------- | ----------------------------------- |
| Per-pixel geolocation, on disk      | 39.3 MB   | 340 GB                              |
| Radiance, counts and masks, on disk | 8.5 MB    | 73 GB                               |
| Whole product, on disk              | 47.7 MB   | 412 GB                              |
| Geometry compute at chunk 10        | 1.06 s    | 2.5 h (parallelizable in principle) |
| NetCDF write of the geolocation set | 1.04 s    | 2.5 h (serial through the client)   |

End to end on the production pairing (`netcdf4` + `distributed`, 2 workers, this laptop), scaled
fixtures ran at 2.57 s/frame at 30 frames, 3.75 s at 60 and 3.35 s at 120, writing 46 to 47 MB
per frame. Extrapolated at 3.4 s/frame a 12 h granule is around 8 h of wall time with two
workers, so more workers are worth having for the compute half even though the write stays
serial.

The nine per-pixel geolocation variables are 82% of the product. They carry `shuffle: true` in
the product definition: byte shuffling ahead of the mandatory gzip level 4 makes the granule
1.6x smaller and the run 1.4x faster (225 MB and 16.3 s become 144 MB and 11.9 s on
`DITL_3min`), and every one of the nine reads back bit-identical. The three
`Terrain_Corrected_*` fields carry it too, so each pair is encoded alike once `LIBSDC-814` gives
them real values; they hold constant fill today, where the filter costs about 0.03 MB/frame
each.

The setting is pinned on the two large non-geolocation fields as well, because the engines
disagree on the default: netCDF4-python turns shuffle on whenever zlib is on, while h5py leaves
it off. Left undeclared, `Radiance` was shuffled under the production engine and not under the
default one, at 4.21 against 2.35 MB/frame, which is 16 GB over a 12 h granule for a filter that
hurts it. `Radiance` is therefore declared `shuffle: false`, and `Pixel_Counts`, which the
filter helps (1.88 to 1.60 MB/frame), `shuffle: true`. With both pinned the two engines agree to
within 0.1 MB/frame on the same data. `Camera_Mask` and `Integration_Time_Flag` are `uint8`,
where shuffling is a no-op.

One engine difference remains that the product definition cannot pin. The engines choose
different chunk shapes, and netCDF4-python's grow along time — `(12, 512, 512)` at 60 frames
against h5py's `(2, 128, 128)` — which compresses the geometry fields worse: a 60-frame granule
is 2929 MB under `netcdf4` against 2451 MB under `h5netcdf`, about 56 GB over a 12 h granule
once the `Radiance` difference above is removed. Pinning `chunksizes` would settle it, which is
what the `chunksizes` bug below blocks, so that upstream fix is worth considerably more than its
3% direct gain.

Two encoding knobs are not usable from the product definition. `complevel` is overwritten by
`libera_utils.io.product_definition.DEFAULT_ENCODING` (gzip 4 is the SDC's setting; level 9
would buy a further 4%), and `chunksizes` is rejected by the h5netcdf engine because YAML
parses it to a list where h5py requires a tuple. Chunk shape is therefore left to the engine,
which costs about 3% against 512x512 chunks.

### Granule length is capped by client memory

Computed blocks are not released as they are written, so whole-run peak memory in the client
grows with granule length under both write paths. Measured on scaled DITL fixtures at the
default chunk of 10:

| Frames | Production: `netcdf4` + `distributed`, 2 workers | Default: `h5netcdf` + `synchronous` |
| ------ | ------------------------------------------------ | ----------------------------------- |
| 3      | —                                                | 2.9 GB                              |
| 30     | 1.9 GB                                           | 6.4 GB                              |
| 60     | 3.0 GB                                           | 9.0 GB                              |
| 120    | 5.7 GB                                           | —                                   |

Least-squares fits of those points:

$$
\text{client peak RSS} \approx 0.6\,\text{GB} + 42\,\text{MB} \times N_{\text{frames}}
\quad\text{(production)}
$$

$$
\text{client peak RSS} \approx 2.8\,\text{GB} + 106\,\text{MB} \times N_{\text{frames}}
\quad\text{(default)}
$$

The production pairing is 2.5x better per frame, because the geometry blocks are computed in
workers rather than in the client, but it is still linear rather than bounded. Under
`distributed` the workers hold their own memory on top of this, one geometry task each, so size
a container as

$$\text{container} \ge \text{client}(N_{\text{frames}}) + \text{DASK\_NUM\_WORKERS} \times (2.2\,\text{GB} + 0.15\,\text{GB} \times \text{LIBERA\_CAM\_GEO\_CHUNK\_SIZE})$$

Neither path responds to `LIBERA_CAM_GEO_CHUNK_SIZE` in its client term (60 frames peak at
9.0 GB with a chunk of 10 and 8.6 GB with a chunk of 2 under the default path). Stage
checkpoints put all of the growth inside `to_netcdf`: at 60 frames the dataset is still lazy at
0.44 GB after both `enforce_dataset_conformance` and `check_dataset_conformance`.

So on the production pairing, the client term alone allows roughly 750 frames (about an hour of
data at 5 s cadence) in 32 GB and 1500 frames (two hours) in 64 GB, before subtracting worker
memory. A 12 h granule would need roughly 360 GB in the client and is not writable on either
path at any chunk size or compression setting. Nothing enforces the cap today: a run past it is
killed by the container rather than stopped by the pipeline. Releasing blocks as they are
written (roadmap item 5) is the prerequisite for full-length granules.

### Examples

Single-process run (default, suitable for tests and small products):

```bash
export PROCESSING_PATH=/path/to/output
export DASK_SCHEDULER=synchronous
libera-cam /path/to/input_manifest.json
```

Multi-worker local cluster. This currently fails when the product is written (see the tuning
guide, item 1); it is recorded for when the write path is fixed:

```bash
export PROCESSING_PATH=/path/to/output
export DASK_SCHEDULER=distributed
export DASK_NUM_WORKERS=4
export DASK_MEMORY_LIMIT=8GB
export LIBERA_CAM_CHUNK_SIZE=50
export LIBERA_CAM_GEO_CHUNK_SIZE=10
libera-cam /path/to/input_manifest.json
```

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
