# Version Changes

## 0.3.0

- **L1A consumption:** Require image-centric WFOV L1A from libera-utils ≥ 5.10.5
  (`CAMERA_TIME`, `WFOV_COMPRESSED_IMAGE`). Remove `reassemble_image_blobs` and the FSW/FPGA
  header parser; JPEG-LS decompression operates on pre-sliced blob payloads. Drop images whose
  `WFOV_HEADER_PARSE_VALID` is false.
- **Exposure metadata:** Convert FPGA actual exposure registers and `WFOV_IMAGE_HEADER_DELTA` to
  milliseconds; write `Actual_Exposure_Time_1/2` and `Exposure_Delta` on L1B.
- **Azimuth valid range:** Correct `Azimuth` `valid_range` to `[-180, 180]`; the FSW field is a
  signed angle in degrees, not a 0-360 bearing.
- **L1B product definition:** Rename misnamed operational-mode fields to
  `Radiometer_Observation_ID` / `Camera_Observation_ID`; add `Image_Mode` and
  `Camera_Packet_Index`.
- **Fixtures / docs:** Replace DITL L1A fixture with enhanced product; scale fixtures along
  `CAMERA_TIME`. Document exposure timing equations in
  `doc/wfov_fsw_header_reference.md`.

## 0.2.6

- **Process Parallelism**: Tested with Dask schedulers; **`synchronous`** (default) and **`distributed`** work reliably. **`threads`** and **`processes`** are rejected at runtime because CSPICE/SPICE is not thread-safe within a worker process.
- **Tuning & Ingestion**: Exposed chunk size configuration via `LIBERA_CAM_CHUNK_SIZE` (default 50) to optimize for specific compute environments. Batch JPEG-LS decompression now builds pre-chunked Dask arrays at ingestion. Added operator configuration guide and tuning relationships in [doc/overview.md](overview.md).
- **Geolocation Memory Optimization**: Switched worker geolocation calculations to `float32` preallocated arrays in `libera_cam/geolocation.py`, eliminating duplicate array copies from `np.stack` and reducing worker live memory peak by ~4×.
- **Dependencies**: Added `netCDF4>=1.6.0` and `distributed>=2026.1.1` to package dependencies for distributed execution and NetCDF file inspection.
- Replace `no_geo` manifest key with `use_geo` (default true; `use_geo: false` for ground-calibration placeholder geolocation). Reject incompatible `use_geo: false` + `jpss_only: true` combinations. Align `use_geo: false` placeholder geolocation with product `_FillValue` (-999 lat/lon, -9999 alt) and Azimuth -999.
- Add `jpss_only` manifest configuration: load only JPSS-SPK and JPSS-CK dynamic kernels, compute per-pixel geolocation using `wfov_pixel_vectors.npy` with `LIBERA_BASE` reference frame (zero-azimuth approximation), and write Azimuth as 0°. Validate required SPICE data products in `read_all_input_data` (production: AZROT-CK + JPSS-SPK + JPSS-CK; jpss_only: JPSS-SPK + JPSS-CK). Reject duplicate kernel types in the manifest. Warn when other SPICE files are listed but skipped. Apply product `_FillValue` for off-Earth and masked pixels; fix Altitude units metadata to meters.

## 0.2.4

- Set `algorithm_version` dynamically from the installed package at write time (product definition YAML uses `null`, matching `libera_rad`). Added tests to ensure the bundled product definition stays in sync with the repo version.
- Replace `DITL_short` integration fixtures with a smaller `DITL_3min` dataset (~3 minutes of WFOV L1A and SPICE kernels). Add a shared `generate_input_manifest` pytest fixture for manifest-driven integration tests.
- Add `doc/wfov_fsw_header_reference.md`, documenting the 36-byte FSW metadata block, `img_mode` semantics, and guidance for separating VIDEO double-image pairs that share duplicate `CAMERA_TIME` timestamps. Cross-reference from `metadata_parser.py`.

## 0.2.3

- Use `KernelManager.load_libera_dynamic_kernels(...)` with libera_utils `KernelFileCache` rather than copying `.bc`/`.bsp` kernels into a package-local directory during L1B processing. Dynamic kernels are provided as an explicit **sequence** of sources (e.g. manifest order); `GeolocationKernelConfig` and integration tests pass ordered `.bc`/`.bsp` source lists.

## 0.2.2

- **Production-Ready Architecture**: Transitioned the entire L1B processing pipeline to a fully lazy, memory-efficient execution model using Dask. This is a step towards processing of full-day science products (~3TB uncompressed) on standard compute nodes without OOM errors.
- **Robust Integration**: Unified L1A packet ingestion, radiometric calibration, and SPICE-based geolocation into a coherent, thread-safe pipeline that rigorously adheres to the L1B Product Definition.
- **Lazy Execution**: Implemented strict Dask-based lazy evaluation for the entire pipeline to handle large daily volumes (~3TB/day) within memory limits. Removed eager `.load()` calls during data ingestion.
- **Dask-Optimized Geolocation**: Integrated `libera_cam.geolocation` with Dask `map_blocks` to parallelize SPICE calculations while managing kernel loading safely on workers.
- **Process Parallelism**: Enforced `synchronous` or `processes` scheduling to ensure thread-safety for CSPICE operations.
- **Vectorized Calibration**: Refactored `convert_dn_to_radiance` to use `xr.apply_ufunc` for truly lazy, vectorized operations on Dask arrays.
- **Product Packaging**: Decoupled product formatting logic into `libera_cam.packaging` to enforce strict adherence to L1B Product Definition (renaming, transposing, typing) transparently.
- **Tuning**: Exposed chunk size configuration via `LIBERA_CAM_CHUNK_SIZE` (default 50) to optimize for specific compute environments.
- **Test Refactoring**: Rewrote `tests/unit/test_l1b.py` and `tests/unit/test_camera.py` to decouple them from legacy data files and non-linearity logic. Used rigorous mocking for orchestration tests.
- **Integration Stability**: Updated integration tests to use `synchronous` Dask scheduling to avoid CSPICE kernel conflicts during parallel test execution.
- **Cleanup**: Removed unused test data files (`camera_calibration_data.h5`) and obsolete code related to non-linearity corrections.

## 0.2.1

- Added `add_geolocation_to_dataset` for Dask-based lazy geolocation computation.
- Optimized geolocation memory usage by moving `pointing_vectors` loading to Dask workers via `mmap`, avoiding massive serialization overhead.
- Introduced `GeolocationKernelConfig` to safely configure SPICE kernel managers on Dask workers.
- Added support for both static (2D) and dynamic (3D) pixel masking in geolocation calculations to skip processing of invalid/dark pixels per-timestamp.
- Added explicit `is_dynamic_mask` configuration to replace brittle dimension-based detection.
- Fixed a bug in `calculate_all_pixel_lat_lon_altitude` where static mask results were inconsistently reshaped and assigned.
- Implemented performance assertions in integration benchmarks to detect processing time regressions.
- Refined dynamic mask loop to process frames serially within workers to ensure SPICE thread-safety while maintaining chunk-level parallelization.
- Added logic to correctly align Time dimensions during Dask `map_blocks` execution when using 3D masks.
- Added `valid_pixel_mask` variable to the L1A dataset to identify valid data pixels (value > 0).

## 0.2.0

- Created memory-efficient L1A image parsing in `libera_cam/image_parsing` using Dask for lazy execution.
- Implemented robust L1A packet stitching with a generator-based state machine and validation for offset continuity and SOP/EOP flags.
- Added unit and integration tests for L1A parsing, including handling of corrupted packet streams.
- Integrated `stitching_stats` into final Dataset global attributes for quality reporting.
- Refactored `read_l1a_cam_data` to return a lazy Xarray Dataset and utilize mission-wide constants for image dimensions.
- Enhanced image processing diagnostics using `logger.exception` for unexpected failures.
- Improved handling of incomplete images by gracefully discarding corrupted partial blobs instead of crashing.

## 0.1.3

- Added first draft of geolocation calculations

## 0.1.2

- Add draft product definition for the L1B camera product to support writing output files during algorithm and pipeline
  testing
