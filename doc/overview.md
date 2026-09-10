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

| Variable                    | Default       | Description                                                                                                                                                                                                       |
| --------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `XARRAY_NETCDF_ENGINE`      | `h5netcdf`    | NetCDF engine used by `libera_utils` to write the product. Both engines write to S3 and both work under `distributed`; `h5netcdf` is faster and produces smaller granules (see the tuning guide).                 |
| `DASK_SCHEDULER`            | `synchronous` | Dask scheduler. **`synchronous`** (single-process, default) or **`distributed`** (local cluster with dashboard). **`threads`** and **`processes`** are not supported — CSPICE is not thread-safe within a worker. |
| `DASK_NUM_WORKERS`          | `1`           | Number of Dask workers when using `distributed`.                                                                                                                                                                  |
| `DASK_MEMORY_LIMIT`         | `8GB`         | Per-worker memory limit for `distributed` (e.g. `4GB`, `8GB`, `16GB`).                                                                                                                                            |
| `LIBERA_CAM_CHUNK_SIZE`     | `50`          | Number of L1A images per Dask batch during JPEG-LS decompression in `read_l1a_cam_data`. Lower values reduce peak memory; higher values reduce scheduler overhead.                                                |
| `LIBERA_CAM_GEO_CHUNK_SIZE` | `10`          | Number of frames per per-pixel geometry task in `add_geolocation_to_dataset`, independent of the decompression chunk. Each task furnishes the kernels once and geolocates its frames one at a time.               |

### Operator Tuning Guide

When sizing AWS Batch containers and configuring Dask execution, keep the following relationships in mind:

1. **Production Scheduling**: `DASK_SCHEDULER=distributed` works with either engine, and `h5netcdf` is the better of the two: at 60 frames it ran in 176 s against 217 s for `netcdf4` and wrote 40.5 MB/frame against 47.3 MB/frame. Earlier releases required `netcdf4` under `distributed` because the `h5netcdf` branch of `libera_utils.io.netcdf.write_libera_data_product` handed `to_netcdf` an open file object, which the distributed scheduler cannot serialize to its workers (`TypeError: cannot pickle '_io.BufferedRandom' object`). `libera_utils` always passes a path from 5.10.9, staging cloud destinations on local disk and uploading them, so both engines write to S3 and both pickle. This repo still pins `libera-utils <5.10.3`, so the pin has to move before a run here can rely on it. `xarray` wraps that path in a `CachingFileManager` and each worker reopens the file itself, which means every worker must see the same filesystem — automatic on the single-node `LocalCluster` built here, but a constraint on any multi-node cluster. Prefer `h5netcdf`: one `netcdf4` + `distributed` run in three failed with `KeyError` inside `netCDF4_.get_array`, a worker having reopened the output before the client's variable definitions were visible to it. Roadmap item 7 carries the comparison. Measured with dask/distributed 2026.8.0, xarray 2026.7.0, h5netcdf 1.8.1, netCDF4 1.7.2.
2. **Parallelism is Capped by Task Count**: Decompression runs in `ceil(N_images / LIBERA_CAM_CHUNK_SIZE)` tasks and per-pixel geometry in `ceil(N_images / LIBERA_CAM_GEO_CHUNK_SIZE)` tasks; inside a geometry task, frames are evaluated serially for SPICE thread-safety. The maximum number of useful workers for each stage is its task count:
   $$\text{max\_useful\_workers} = \left\lceil \frac{N_{\text{images}}}{\text{chunk size}} \right\rceil$$
   For example, a 500-image product with `LIBERA_CAM_GEO_CHUNK_SIZE=10` creates 50 geometry tasks and can utilize at most 50 workers on that stage. Smaller chunks add per-task kernel furnishing and scheduler overhead.
3. **Task Memory Sizing**: Each active worker, or the single process under `synchronous`, holds one task at a time. A geometry task holds its output block, 143 MB per frame of eight `float32` fields and the `uint16` flags, on top of the kernels, pixel vectors and the curryer transients for the frame in progress. Measured by calling `calculate_chunk_geometry` directly in a fresh process, one task of `N` frames peaks at 2.21, 2.51, 2.99, 3.74 and 5.19 GB for `N` of 1, 2, 5, 10 and 20, a least-squares fit of

   $$\text{task peak RSS} \approx 2.2\,\text{GB} + 0.15\,\text{GB} \times N_{\text{frames in the task}}$$

   Per-frame cost falls from 2.08 s at one frame to 1.00 s asymptotically, so the ~1.1 s of per-task kernel furnishing is 6% overhead at the default chunk of 10 and 17% at 5. A decompression task at `LIBERA_CAM_CHUNK_SIZE=50` holds ~800 MB of `image_data` and masks. This bounds one task, not one run: whole-run peak memory is set by the write path instead, and is the subject of "Memory is bounded; write throughput is not" below.

4. **Container Sizing Uses the Whole Process Tree**: Computed blocks accumulate for the length of a write, not the length of the granule, so peak memory plateaus rather than growing with frame count. Under `synchronous` that plateau sits in the client; under `distributed` the store tasks run on the workers, so a client-only measurement misses most of it. Size against the whole tree; see "Memory is bounded; write throughput is not" below.

### Data volume and throughput

Measured on the `DITL_3min` granule (three 2048x2048 frames, 21.7% of pixels off the
ellipsoid), single core, local SSD, with the default `h5netcdf` engine:

| Quantity                            | Per frame | 12 h granule at 5 s (8640 frames)    |
| ----------------------------------- | --------- | ------------------------------------ |
| Per-pixel geolocation, on disk      | 39.3 MB   | 340 GB                               |
| Radiance, counts and masks, on disk | 8.5 MB    | 73 GB                                |
| Whole product, on disk              | 47.7 MB   | 412 GB                               |
| Geometry compute at chunk 10        | 1.06 s    | 2.5 h (parallelizable in principle)  |
| NetCDF write of the geolocation set | 1.04 s    | 2.5 h (serialized by the write lock) |

End to end on scaled fixtures (one worker, this laptop, `DASK_MEMORY_LIMIT=24GB`), `h5netcdf`
ran at 2.97 s/frame at 30 frames and 2.93 s at 60 under `distributed`, and 3.03 s and 2.97 s
under `synchronous`, writing 40.5 MB/frame throughout; `netcdf4` + `distributed` ran at
3.61 s/frame at 60 frames and wrote 47.3 MB/frame. Extrapolated at 2.93 s/frame a 12 h granule
is around 7 h of wall time on one worker. Scheduler choice buys almost nothing at one worker,
as expected — its value is in adding workers to the compute half, which the write lock does not
serialize.

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

### Memory is bounded; write throughput is not

Sizing a container needs the memory of the **whole process tree**, not the client: under
`distributed` the workers are separate processes, and the client figure omits them entirely.
Peak RSS sampled across the client and all descendants, scaled DITL fixtures and two real
granules, at the default chunk of 10:

| Frames      | Pairing                    | Client only | Whole tree |
| ----------- | -------------------------- | ----------- | ---------- |
| 30          | `h5netcdf` + `synchronous` | 8.50 GB     | 8.50 GB    |
| 60          | `h5netcdf` + `synchronous` | 9.65 GB     | 9.65 GB    |
| 30          | `h5netcdf` + `distributed` | 1.86 GB     | 10.66 GB   |
| 60          | `h5netcdf` + `distributed` | 2.60 GB     | 8.21 GB    |
| 60          | `netcdf4` + `distributed`  | 3.07 GB     | 8.07 GB    |
| 160 (real)  | `h5netcdf` + `distributed` | 6.99 GB     | 9.14 GB    |
| 1003 (real) | `h5netcdf` + `distributed` | 7.22 GB     | 7.38 GB    |

**Memory does not grow with granule length.** A 1003-frame granule peaks _lower_ than a
60-frame one. The per-frame memory models in earlier releases of this document were an artefact
of measuring `ru_maxrss` on `RUSAGE_SELF`, which does not count `distributed`'s workers, and of
reading run-to-run variation in the terminal flush as a trend. The RSS trace shows why: memory
ramps to a plateau in the first minute and stays there — 5.6 GB across the 160-frame run, 4.5 GB
across the 1003-frame run — with a short spike at the end. Blocks are held for the length of a
write, not for the length of the granule. The "roughly 750 frames in 32 GB" cap that earlier
releases derived is withdrawn: no such cap was observed.

`DASK_MEMORY_LIMIT` must still be raised above its `8GB` default to run `distributed` at these
sizes: at the default, both `h5netcdf` + `distributed` and `netcdf4` + `distributed` died with
`distributed.scheduler.KilledWorker` on the store tasks at 60 frames. Every measurement above
used `DASK_MEMORY_LIMIT=24GB`. Peak is insensitive to `LIBERA_CAM_GEO_CHUNK_SIZE` (60 frames
peak at 9.0 GB with a chunk of 10 and 8.6 GB with a chunk of 2 under the default path), and
stage checkpoints put the growth inside `to_netcdf`: at 60 frames the dataset is still lazy at
0.44 GB after both `enforce_dataset_conformance` and `check_dataset_conformance`.

What _does_ degrade with granule length is throughput:

| Granule           | Frames | Wall     | s/frame  | MB/frame |
| ----------------- | ------ | -------- | -------- | -------- |
| 15 Minute July 12 | 160    | 7.9 min  | **2.96** | 43.0     |
| 3 Hour July 11    | 1003   | 85.9 min | **5.14** | 31.4     |

That is 1.74x slower per frame over a 6x longer granule, so a 12 h granule extrapolates to well
over the 7 h that the 160-frame rate implies. The leading explanation is HDF5 chunk
misalignment. `h5py.guess_chunk` is shape-dependent, so the chunk changes with granule length —
`(2, 128, 128)` at 60 frames, `(5, 64, 128)` at 160, `(16, 64, 64)` at 1006 — while Dask writes
geolocation in `LIBERA_CAM_GEO_CHUNK_SIZE`-frame blocks. Ten frames divides evenly into a depth
of 2 and of 5 but not of 16, so at production sizes every write lands on partial HDF5 chunks and
the library decompresses, modifies, recompresses and rewrites them. A `sample` of the worker
during the 1003-frame run has `H5Z__filter_shuffle` as the dominant leaf, and the file stops
growing during those phases because rewriting an existing chunk in place adds no bytes. The
spatial tile also shrinks to 64x64, giving 838,656 chunks across the 13 3-D variables.

This has not been confirmed by re-running with a pinned chunk shape, because `chunksizes` cannot
currently be set from the product definition (see the encoding note above). That upstream fix is
worth considerably more than the 3% direct compression gain quoted there: it converts an
unpredictable, granule-length-dependent write cost into a fixed one.

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
