# libera_cam L1B Performance

The measurements behind the L1B Dask defaults and the product encoding.

**Related documentation**

- [overview.md](overview.md) — running the pipeline and its production settings
- [roadmap.md](roadmap.md) — forward-looking L1B plans
- [CHANGELOG.md](../CHANGELOG.md) — shipped changes by version

---

## Tuning relationships

When sizing AWS Batch containers and configuring Dask execution, keep the following relationships in mind:

1. **Production Scheduling**: `DASK_SCHEDULER=distributed` writes through worker compression (see [AWS sizing](#aws-sizing)) and requires `h5netcdf`; libera-utils raises `ValueError` if `XARRAY_NETCDF_ENGINE` is `netcdf4`. `h5netcdf` was also the better engine with HDF5 compressing in the writing process: at 60 frames it ran in 176 s against 217 s for `netcdf4` and wrote 40.5 MB/frame against 47.3 MB/frame. The client process alone writes the output; each worker reads the L1A file by path (`_decompress_batch_from_source`, `kept_pixels_from_source`), which means every worker must see the client's filesystem — automatic on the single-node `LocalCluster` built here, but a constraint on any multi-node cluster. Measured with dask/distributed 2026.8.0, xarray 2026.7.0, h5netcdf 1.8.1, netCDF4 1.7.2.
2. **Parallelism is Capped by Task Count**: Decompression runs in `ceil(N_images / LIBERA_CAM_CHUNK_SIZE)` tasks and per-pixel geometry in `ceil(N_images / LIBERA_CAM_GEO_CHUNK_SIZE)` tasks; inside a geometry task, frames are evaluated serially for SPICE thread-safety. The maximum number of useful workers for each stage is its task count:
   $$\text{max\_useful\_workers} = \left\lceil \frac{N_{\text{images}}}{\text{chunk size}} \right\rceil$$
   For example, a 500-image product with `LIBERA_CAM_GEO_CHUNK_SIZE=10` creates 50 geometry tasks and can utilize at most 50 workers on that stage. Smaller chunks add per-task kernel furnishing and scheduler overhead.
3. **Task Memory Sizing**: Each active worker, or the single process under `synchronous`, holds one task at a time. A geometry task holds its output block, 151 MB per frame of eight `float32` fields and the `uint32` flags, on top of the kernels, pixel vectors and the curryer transients for the frame in progress. Measured by calling `calculate_chunk_geometry` directly in a fresh process, one task of `N` frames peaks at 2.21, 2.51, 2.99, 3.74 and 5.19 GB for `N` of 1, 2, 5, 10 and 20, a least-squares fit of

   $$\text{task peak RSS} \approx 2.2\,\text{GB} + 0.15\,\text{GB} \times N_{\text{frames in the task}}$$

   Per-frame cost falls from 2.08 s at one frame to 1.00 s asymptotically, so the ~1.1 s of per-task kernel furnishing is 6% overhead at a chunk of 10 and 17% at the default of 5. With HDF5 compressing in the writing process the run was write-bound, so on AWS that overhead did not show in wall time (see [AWS sizing](#aws-sizing)). A decompression task at `LIBERA_CAM_CHUNK_SIZE=50` holds ~800 MB of `image_data` and masks. This bounds one task, not one run: whole-run peak memory is set by the write path instead (see [Memory](#memory)).

4. **Geometry Tasks Take No Dask Inputs**: `distributed` holds a task with no inputs back until a worker has a free thread, and with one thread per worker that thread is busy with the blocks already started: their computation and, under `distributed`, their compression. That back-pressure is what keeps the per-pixel geometry, 151 MB per frame of output, from piling up ahead of them. Each geometry task therefore rebuilds its frames' kept-pixel masks from the L1A file itself (`kept_pixels_from_source`, 0.025 s/frame) instead of depending on the `image_data` or mask tasks. Fed from `image_data`, the full 695-frame DITL2 granule restarted workers and spilled 60-75 GB to disk on AWS; input-free, it ran in 18.6 min on three 12 GB workers with no restart and no spill, each worker holding one geometry block at a time. A change that gives the geometry tasks an input brings the backlog back.
5. **Container Sizing Uses the Whole Process Tree**: Computed blocks accumulate for the length of a write, not the length of the granule, so peak memory plateaus rather than growing with frame count (measured with HDF5 compressing in the writing process). Under `synchronous` that plateau sits in the client; under `distributed` the workers compute and compress the blocks and hold each result until the client stores it, so a client-only measurement misses most of it. Size against the whole tree; see [Memory](#memory).

## AWS sizing

Measured on the 695-frame DITL2 granule as AWS Batch Fargate Spot tasks, `h5netcdf` +
`distributed`, with pinned chunks and input-free geometry tasks, and with HDF5 compressing in the
writing process:

| Task           | Workers x limit | Decomp chunk | Geo chunk | Wall           | Task peak   | Worker pauses / restarts |
| -------------- | --------------- | ------------ | --------- | -------------- | ----------- | ------------------------ |
| 8 vCPU / 60 GB | 5 x 12 GB       | 50           | 10        | 25.2 min       | 19.7 GB     | 0 / 0                    |
| 4 vCPU / 30 GB | 3 x 9 GB        | 50           | 10        | 18.4 min       | 9.1 GB      | 13 / 2                   |
| 4 vCPU / 30 GB | 3 x 9 GB        | 20           | 10        | 19.3, 26.6 min | 9.0, 9.4 GB | 0 / 0                    |
| 4 vCPU / 30 GB | 3 x 9 GB        | 20           | 5         | 19.5 min       | 5.1 GB      | 0 / 0                    |
| 2 vCPU / 16 GB | 2 x 6 GB        | 10           | 10        | 35.7 min       | 6.7 GB      | 7 / 2                    |
| 2 vCPU / 16 GB | 2 x 6 GB        | 10           | 5         | 22.9, 21.4 min | 3.6, 3.4 GB | 0 / 0                    |
| 4 vCPU / 16 GB | 3 x 4.5 GB      | 20           | 10        | failed         | 6.7 GB      | 14 / 8, `KilledWorker`   |

The default chunks, 20 and 5, come from the 4 vCPU / 30 GB rows. With HDF5 compressing in the
writing process, the smallest setting that ran cleanly was 2 vCPU / 16 GB with two 6 GB workers
and a decompression chunk of 10; it also ran a full day of July 12 as four 6-hour granules of
about 2,050 frames, each in 59-66 min at a 3.7-3.9 GB peak. In those runs HDF5 compressed every chunk in the single writing process, which was about 95%
of wall time while CPU averaged 1.3-2.0 vCPU, so workers, memory and chunks changed memory and
stability, not speed. Spot hosts varied wall time by about 40% on identical settings. Ephemeral
disk peaks near the product size plus about 3-4 GB: 8-9 GB at 695 frames, 16-19 GB at about 2,050.

Under `distributed` the workers compress the chunks of every Dask-backed three-dimensional
variable and the writing process only stores them (libera-utils `worker_compressed_variables`).
The full July 12 day, 8,205 frames in one file, ran through the l1b-cam step function on on-demand
Fargate at 4 vCPU / 30 GB with three 9 GB workers and chunks of 20 and 5:

| Write path                 | Write and upload | Per frame | Step function | Task peak    | Product  |
| -------------------------- | ---------------- | --------- | ------------- | ------------ | -------- |
| HDF5 compresses            | about 6 h 11 min | 2.7 s     | 6 h 17 min    | not recorded | 57.86 GB |
| Workers compress, run 1    | 1 h 37 min       | 0.69 s    | 1 h 47 min    | not recorded | 57.31 GB |
| Workers compress, runs 2-4 | 2 h 32-33 min    | 1.09 s    | 2 h 42-43 min | 10.7-10.8 GB | 57.31 GB |

Per frame is the write alone, without the upload of about 3.7 min. Runs 2-4 are the typical
time: their writes, over 18 hours, two with Container Insights on and one with it off, were
within 32 s of each other. With Insights on, memory peaked at 10.7-10.8 GB of the task's 30 GB
and averaged 9.8-9.9 GB, ephemeral disk peaked at 62.8 GB, and CPU averaged 2.8-2.9 of 4 vCPU.
Run 1's write took 37% less time with the same image, L1A, kernels and 1,607,052 stored chunks;
Fargate does not fix the host CPU model, so it most likely ran on a faster host. No worker run
logged a worker pause, restart or spill. The worker-compress runs skipped chunks that were fill
throughout, most of them in the `Terrain_Corrected_*` placeholders, so their product is 0.55 GB
smaller; libera-utils 5.17.0 stores every chunk, as `to_netcdf` does. Locally, on 150 frames of
the same day, the two write paths produced bit-identical variables.

With the workers compressing, the write scales with vCPUs. On the DITL2 granule as on-demand
Fargate tasks, with chunks of 20 and 5, two runs each:

| Task             | Workers x limit | Write per frame | CPU average | Task peak |
| ---------------- | --------------- | --------------- | ----------- | --------- |
| 4 vCPU / 30 GB   | 3 x 9 GB        | 1.06, 1.07 s    | 2.9 vCPU    | 3.6 GB    |
| 4 vCPU / 30 GB   | 4 x 6 GB        | 1.05, 1.02 s    | 3.6 vCPU    | 4.9 GB    |
| 8 vCPU / 60 GB   | 6 x 9 GB        | 0.53, 0.62 s    | 5.5 vCPU    | 6.4 GB    |
| 8 vCPU / 60 GB   | 7 x 8 GB        | 0.52, 0.51 s    | 6.0 vCPU    | 8.0 GB    |
| 16 vCPU / 120 GB | 12 x 9 GB       | 0.30, 0.29 s    | 10.6 vCPU   | 12.2 GB   |

The defaults are the 8 vCPU / 60 GB row with seven 8 GB workers: half the write time of 4 vCPU
at the same Fargate cost per day, about $0.70 for a full day's write, and within a compute
environment capped at 10 vCPU. At 4 vCPU a fourth worker gains 3%, because the CPU is already
full. From 4 to 16 vCPU the write runs 3.6 times faster, so the single writing process is not yet
the limit. The 3 x 9 GB rows match the full day's 1.09 s per frame. No run logged a worker pause,
restart or spill.

At the defaults, the full July 12 day wrote in 1 h 20 min against 2 h 28 min, 0.58 s/frame, as a
direct Batch job of 1 h 29 min. It stored the same 1,607,052 chunks in the same 57.31 GB product,
peaked at 15.8 GB of the task's 60 GB and 62.8 GB of disk, averaged 6.2 of 8 vCPU, and logged no
worker pause, restart or spill.

## Data volume and throughput

Only the pixels the camera's onboard pixel mask kept are geolocated; the rest of each frame is
written as NaN, which compresses to almost nothing. Measured on 50 consecutive DITL2 frames
(9.1% of pixels kept), single core, with the product encoding:

| Quantity                                  | Every pixel | Kept pixels only |
| ----------------------------------------- | ----------- | ---------------- |
| Geometry compute                          | 0.95 s      | 0.095 s          |
| NetCDF write                              | 2.04 s      | 1.16 s           |
| Geolocation, radiance and counts, on disk | 41.3 MB     | 7.1 MB           |

Geometry is computed at each kept pixel, not interpolated: interpolating from a subsampled grid
put errors of up to 1.7 degrees near the limb.

Video mode applies no onboard mask, so its frames cost what the "every pixel" column shows. Every
pixel, as in video mode, on the `DITL_3min` granule (three 2048x2048 frames, 21.7% of pixels off
the ellipsoid), single core, local SSD, with the default `h5netcdf` engine:

| Quantity                            | Per frame | 12 h granule at 5 s (8640 frames) |
| ----------------------------------- | --------- | --------------------------------- |
| Per-pixel geolocation, on disk      | 39.3 MB   | 340 GB                            |
| Radiance, counts and masks, on disk | 8.5 MB    | 73 GB                             |
| Whole product, on disk              | 47.7 MB   | 412 GB                            |

The nine per-pixel geolocation variables are 82% of the product. They carry `shuffle: true` in
the product definition: byte shuffling ahead of the mandatory gzip level 4 makes the granule
1.6x smaller and the run 1.4x faster (225 MB and 16.3 s become 144 MB and 11.9 s on
`DITL_3min`), and every one of the nine reads back bit-identical. The three
`Terrain_Corrected_*` fields carry it too, so each pair is encoded alike; they are NaN
throughout, where the filter costs about 0.03 MB/frame each.

The setting is pinned on the two large non-geolocation fields as well, because the engines
disagree on the default: netCDF4-python turns shuffle on whenever zlib is on, while h5py leaves
it off. Left undeclared, `Radiance` was shuffled under `netcdf4` and not under `h5netcdf`, the
default, at 4.21 against 2.35 MB/frame, which is 16 GB over a 12 h granule for a filter that
hurts it. `Radiance` is therefore declared `shuffle: false`, and `Pixel_Counts`, which the
filter helps (1.88 to 1.60 MB/frame), `shuffle: true`. With both pinned the two engines agree to
within 0.1 MB/frame on the same data. `Camera_Mask` and `Integration_Time_Flag` are `uint8`,
where shuffling is a no-op.

Left to themselves the engines also choose different chunk shapes, and netCDF4-python's grow
along time — `(12, 512, 512)` at 60 frames against h5py's `(2, 128, 128)` — which compresses the
geometry fields worse: a 60-frame granule is 2929 MB under `netcdf4` against 2451 MB under
`h5netcdf`. h5py's guessed chunk also depends on granule length — `(2, 128, 128)` at 60 frames,
`(5, 64, 128)` at 160, `(16, 64, 64)` at 1006 — while Dask writes geolocation in
`LIBERA_CAM_GEO_CHUNK_SIZE`-frame blocks, so at production sizes writes land on partial HDF5
chunks that the library decompresses, modifies, recompresses and rewrites; a `sample` of the
worker shows `H5Z__filter_shuffle` as the dominant leaf. Every three-dimensional variable
therefore declares `chunksizes: [1, 512, 512]`, which libera-utils 5.11.4 converts to the tuple
h5py requires, so both engines write the same chunks. A depth of one image divides any block
size, so the write cost does not depend on granule length: the 1003-frame granule went from 85.9
to 48.6 minutes (1.77x) for 5.4% more file, and with four workers from 5.14 to 2.02 s/frame end
to end (2.54x). `complevel` is not usable from the product definition: it is overwritten by
`libera_utils.io.product_definition.DEFAULT_ENCODING` (gzip 4 is the SDC's setting; level 9
would buy a further 4%).

## Memory

Sizing a container needs the memory of the **whole process tree**, not the client: under
`distributed` the workers are separate processes, and the client figure omits them entirely.
Peak RSS sampled across the client and all descendants, scaled DITL fixtures and two real
granules, at a geometry chunk of 10, `DASK_MEMORY_LIMIT=24GB`, with HDF5 compressing in the
writing process:

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
60-frame one. The RSS trace shows why: memory ramps to a plateau in the first minute and stays
there — 5.6 GB across the 160-frame run, 4.5 GB across the 1003-frame run — with a short spike
at the end. Blocks are held for the length of a write, not for the length of the granule.

With input-free geometry tasks the geometry chunk sets the peak (see [AWS sizing](#aws-sizing)).
Stage checkpoints put the growth inside `to_netcdf`: at 60 frames the dataset is still lazy at
0.44 GB after both `enforce_dataset_conformance` and `check_dataset_conformance`.
