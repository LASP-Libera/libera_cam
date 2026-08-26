# libera_cam L1B Roadmap

High-level plan for WFOV camera L1B processing. This document captures known upstream
dependencies, open design questions, and forward-looking engineering priorities.

**Related documentation**

- [overview.md](overview.md) — L1B pipeline usage, manifest configuration, and Dask parallelization
- [wfov_fsw_header_reference.md](wfov_fsw_header_reference.md) — FSW metadata, `img_mode`, VIDEO
  pairing, and duplicate-timestamp guidance
- [changelog.md](changelog.md) — shipped changes by version

---

## 1. Camera-time L1A with image-aligned chunking

### Problem today

L1A WFOV science products arrive as a **CCSDS mem-dump packet stream** indexed by
`PACKET_ICIE_TIME`, not by image acquisition time. Each complete NAND image blob may span
many packets. At L1B, `read_l1a_cam_data.py` must:

1. Scan the packet stream (`reassemble_image_blobs`)
2. Stitch SOP/EOP sequences into image blobs
3. Decompress JPEG-LS payloads
4. Build a `camera_time`-indexed dataset

Packet-boundary chunking does **not** align with image boundaries. A Dask chunk can start or
end mid-image, which prevents clean parallel decomposition and forces stitching work inside
the L1B algorithm.

See [L1A time coordinates](wfov_fsw_header_reference.md#l1a-time-coordinates) for the
distinction between `PACKET_ICIE_TIME` and `CAMERA_TIME`.

### Target state

Move packet stitching and blob reassembly into the **libera_utils L1A pipeline** (tracked in
the associated `libera_utils` L1A camera ingest update). Future L1A products will be:

- Indexed on **`camera_time`** (or an equivalent per-image coordinate)
- Written in **chunks that contain only complete images**

### Expected L1B impact

With image-aligned L1A chunks, L1B can:

- Skip packet stitching entirely
- Map Dask blocks directly to image groups
- Parallelize radiometric correction and geolocation with fewer graph nodes and less
  scheduler overhead

### Open questions

- Chunk size policy (images per chunk vs. target byte size)
- Whether decoded 12-bit arrays or compressed payloads are stored at L1A
- Ensuring L1A chunk boundaries contain only complete images

---

## 2. VIDEO mode and duplicate `camera_time` values

### Problem today

When the FPGA runs in **VIDEO** mode (`img_mode == 1`), a single camera exposure produces
**two downlinked NAND images** with identical FSW timestamps: a full-frame write (bitmask bypass)
and a masked/stripe write. Both share the same `timestamp_seconds` and `timestamp_subseconds`, so
`camera_time` is **not unique**.

Duplicate time coordinates break standard NetCDF indexing and complicate downstream analysis.
See [Images per timestamp](wfov_fsw_header_reference.md#images-per-timestamp) and
[Separating duplicate-timestamp pairs](wfov_fsw_header_reference.md#separating-duplicate-timestamp-pairs).

### Target architecture

Rather than altering timestamps (which introduces synthetic errors), candidate approaches include:

1. **Auxiliary Time Coordinate**: Configure the dataset along an integer frame index dimension
   and store acquisition time as a non-unique auxiliary coordinate variable.
2. **Stream Separation**: Split VIDEO pairs into distinct logical variables or datasets:
   - **ScienceStill**: Masked member of each VIDEO pair plus nominal non-VIDEO frames.
   - **VideoFull**: Full-frame member only.
3. **Metadata Constant Offsets**: If nominal timing differences exist between modes, document
   them in product metadata attributes rather than modifying coordinates.

### L1B responsibilities

| Step                   | Requirement                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Geolocation input time | Use true acquisition epoch for SPICE lookups                                                                             |
| Product separation     | Split VIDEO pairs into distinct logical streams (e.g. science still vs. video full-frame)                                |
| Validation             | Use packet order, `img_mode`, and content heuristics (`valid_pixel_mask`, `integration_mask`) to confirm pair assignment |

---

## 3. Per-pixel integration time and geolocation time

### Problem today

Each downlinked pixel is encoded as **13 bits** in the JPEG-LS payload:

- **Bits 0–11**: DN value (12-bit science data)
- **Bit 12**: integration-time flag (short vs. long exposure)

The camera acquires **two exposures at different integration times** within a single readout
cycle. Pixels from short and long integrations correspond to **different effective acquisition
times**, even though they share one `camera_time` frame coordinate.

L1B extracts `integration_mask` in `l1a_parser.py` for radiometric calibration, but geolocation
currently evaluates SPICE at the frame-level `camera_time` for all active un-masked pixels.

### Target state

1. Model the short/long integration timing relative to the frame timestamp (exposure start,
   readout order, and FPGA timing offsets)
2. Assign per-pixel or per-integration-group time offsets for SPICE interpolation
3. Apply the correct time when computing lat/lon/alt and surface geometry angles for each pixel

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
