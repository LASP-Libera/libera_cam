# libera_cam L1B Roadmap

High-level plan for WFOV camera L1B processing. This document captures known upstream
dependencies, open design questions, and forward-looking engineering priorities.

**Related documentation**

- [overview.md](overview.md) — L1B pipeline usage, manifest configuration, and Dask parallelization
- [wfov_fsw_header_reference.md](wfov_fsw_header_reference.md) — FSW metadata, `img_mode`, VIDEO
  pairing, and duplicate-timestamp guidance
- [changelog.md](changelog.md) — shipped changes by version

---

## 1. Image-aligned NetCDF chunking

The on-disk NetCDF chunk policy for L1A — images per chunk vs. target byte size — is undecided,
so Dask block sizes cannot be matched to disk chunks. `_extract_jpeg_ls_payloads` reads every
blob into memory up front; lazy per-chunk blob reads depend on that policy.

Open questions:

- Whether the policy targets a fixed image count or a target byte size per chunk
- Guaranteeing L1A chunk boundaries contain only complete images

---

## 2. VIDEO mode and duplicate `CAMERA_TIME` values

### Problem today

When the FPGA runs in **VIDEO** mode (`img_mode == 1`), a single camera trigger produces
**two NAND images** with identical FSW timestamps: a full-frame write (bitmask bypass) and a
masked/stripe write. Both share the same `timestamp_seconds` and `timestamp_subseconds`, so
`CAMERA_TIME` is **not unique**.

Duplicate time coordinates break many NetCDF viewers and complicate any logic that assumes
one row per acquisition instant. See
[Images per timestamp](wfov_fsw_header_reference.md#images-per-timestamp) and
[Separating duplicate-timestamp pairs](wfov_fsw_header_reference.md#separating-duplicate-timestamp-pairs).

### Proposed L1A fix

Introduce a small synthetic sub-microsecond offset (on the order of **1 µs**) within each
duplicate-timestamp pair so that stored coordinates are unique and monotonic in packet
order. The offset is a **storage convenience**, not a physical acquisition-time correction.

Document the offset scheme in the L1A product definition and propagate a flag or auxiliary
coordinate so downstream code can recover the true trigger time.

### L1B responsibilities

| Step                   | Requirement                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Geolocation input time | **Undo** the synthetic offset before SPICE lookups so pointing uses the true trigger time                                          |
| Product separation     | Split VIDEO pairs into distinct logical streams (e.g. science still vs. video full-frame); exact output schema **not yet decided** |
| Validation             | Use packet order, `img_mode`, and content heuristics (`valid_pixel_mask`, `integration_mask`) to confirm pair assignment           |

Candidate stream names from the FSW reference doc: **ScienceStill** (masked member of each
VIDEO pair plus non-VIDEO frames) and **VideoFull** (full-frame member only). The L1B fields
available for pair identification are listed in [overview.md](overview.md#l1a-inputs).

### Open questions

- Whether L1B emits separate products, separate variables within one product, or filters one
  stream at write time
- How to represent the "true" vs. "storage" timestamp in L1B output metadata
- Whether all VIDEO frames in a sequence need splitting or only `img_mode == 1` pairs

---

## 3. SPICE-derived Azimuth

`Azimuth` passes through the FSW-reported commanded azimuth from the L1A header. It is to be
replaced by an azimuth derived from SPICE pointing, alongside the geometric viewing and solar
azimuth angles.

---

## 4. Surface geometry geolocation performance & sparse geometry

### Problem today

Sub-satellite geolocation (lat/lon/alt via SPICE ray–ellipsoid intersection) is computationally
intensive. Adding **surface geometry** angles — solar zenith, viewing zenith, relative azimuth,
and related fields at each pixel — multiplies SPICE work substantially.

### Optimization strategies

| Approach                    | Notes                                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Sparse geometry**         | Restrict geometry calculations strictly to unmasked/active pixels (e.g. within stripes), skipping masked-off areas entirely |
| **Reduce SPICE call count** | Batch by unique timestamps or integration groups before per-pixel expansion                                                 |
| **Coarser geometry grid**   | Compute angles on a subsampled grid and interpolate (with science validation)                                               |
| **Worker-local caching**    | Reuse Sun/spacecraft orientation for all pixels sharing a timestamp                                                         |
| **Vectorized evaluation**   | Evaluate whether curryer vectorized paths or precomputed ephemeris tables help at full camera resolution (2048×2048)        |

---

## 5. Direct worker NetCDF chunk output

### Problem today

Currently, `write_data_product` relies on `to_netcdf` / `write_libera_data_product`, which
funnels all computed Dask blocks back through the client process. Client process memory and I/O
become a bottleneck on large multi-gigabyte datasets.

### Target state

Investigate and prototype direct worker-to-disk or worker-to-S3 NetCDF chunk writes (e.g., using
region-based HDF5/NetCDF-4 writes or Zarr storage targets). This allows distributed workers to
write their completed chunks independently without returning payload data to the client driver.

---

## 6. Automated performance regression benchmarking

### Target state

Integrate `pytest-benchmark` to track execution time and memory scaling for standard camera
chunk sizes across CI runs. This will provide early detection of performance regressions in
radiometric calibration, JPEG-LS decompression, and SPICE geolocation.

---

## 7. Dask performance testing

## Aspects worth testing more thoroughly

- Logging of distributed tasks on AWS. Are we getting enough information out to debug?
- When using `distributed` mode does Dask emit a "large object detected in task graph" warning because of the 50 images passing in?
- Confirm netcdf file writing in distributed mode on AWS with h5netcdf vs netcdf4 engine
