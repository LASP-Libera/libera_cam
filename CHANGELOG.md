# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Populate the spacecraft-level geometry fields from `curryer.compute.geometry.GeometryData`, following the `libera_rad` implementation: `Subsatellite_Latitude`/`_Longitude`/`_Colatitude`, `Subsolar_Latitude`/`_Longitude`/`_Colatitude`, `Radius_of_Satellite_from_Center_of_Earth`, and the granule attribute `Earth_Sun_Distance_AU`, plus the new `Satellite_Position` and `Satellite_Velocity` (J2000, on `EUCLIDEAN_DIM`) and `Satellite_Attitude_Q0..Q3` (ECEF). One `JPSS4_SC` query serves production and `jpss_only` alike. A failed curryer SPICE query raises a parsed, user-facing `RuntimeError`, and a SPICE input these fields read (spacecraft ephemeris, attitude, Sun ephemeris) that covers no frame of the granule raises `ValueError`, through curryer's `require_coverage`.
- Add `Azimuth_Motor_Position_SPICE`, the motor encoder angle from the AZROT-CK (third 1-2-3 Euler angle of `LIBERA_BASE_COORD -> LIBERA_AZ_COORD`, degrees in `[0, 360)`) at `CAMERA_TIME`, via `curryer.compute.spatial.frame_to_frame_euler`, matching `libera_rad`, and the same angle at each integration: `Azimuth_Motor_Position_SPICE_Int_Time_1` at `CAMERA_TIME` and `Azimuth_Motor_Position_SPICE_Int_Time_2` at `CAMERA_TIME` + 350 ms. `jpss_only` writes 0 degrees and `use_geo: false` writes NaN. A granule the azimuth CK covers at no frame raises `ValueError`. Elevation is not reported: the camera is mounted on the azimuth stage and ELSCAN-CK is not a camera L1B input.
- Frames the kernels do not cover are written as NaN in the per-frame variables, and a warning gives their count for each variable. With `use_geo: false` the per-frame variables and `Earth_Sun_Distance_AU` are NaN.
- Add the global attribute `Geolocation_Mode`: `production`, `jpss_only` or `use_geo_false`.
- Geolocate every pixel of every frame through curryer's `spatial.pixel_geometry` (`geolocate_frame`): `Latitude`, `Longitude`, `Altitude` and the five surface angles (`Solar_Zenith_Surface`, `Viewing_Zenith_Surface`, `Solar_Azimuth_Surface_WRT_North`, `Viewing_Azimuth_Surface_WRT_North`, `Relative_Azimuth_Surface`) are now produced per pixel with the product fills, and the per-pixel `Geolocation_Quality_Flag` (uint16; curryer `SpatialQualityFlags` in bits 0-14, bit 15 = geolocation not run) is added to the product definition. Every pixel that hits the ellipsoid is published; `valid_pixel_mask` no longer gates geolocation. Azimuths that round to 360.0 in float32 are wrapped to 0.
- Add the per-pixel `Viewing_Azimuth_Surface_WRT_North` and `Solar_Azimuth_Surface_WRT_North` to the product definition, matching the `libera_rad` surface-angle set, so `Relative_Azimuth_Surface` can be derived from them (curryer's CERES convention).
- Add a SPICE-versus-FSW azimuth diagnostic (`tests/helpers/azimuth_diagnostics.py`, shortest-arc statistics) with an integration test on the DITL data, adapted from the comparison logging in PR #16 (Matt Watwood). Analysis only; the header value is not a production reference.

### Changed

- **Dependencies**: Replaced `pillow-jpls` with `pyjpegls>=1.5.1` for JPEG-LS decompression. `pillow-jpls` publishes wheels only through cp312, so Python 3.13 installs built it from source through Conan and CMake and failed outright on gcc 14 hosts. `pyjpegls` binds the same CharLS codec and ships wheels for cp39–cp313 on manylinux x86_64 and aarch64, macOS, and Windows, so no interpreter in the supported range needs a compiler. Decoded sample values are unchanged.
- **Image decoding**: `read_l1a_cam_data` returns `image_data` as `uint16` (was `int32`), the decoder's native sample type and the L1B `Pixel_Counts` dtype. Values and the L1B product are unchanged.
- Require `lasp-curryer >= 0.5.3` for `GeometryData.get_geometry(require_coverage=)`.
- Require `libera-utils >=5.16.0,<6.0.0`. Since 5.14.0 libera-utils names data products with a microsecond timestamp revision (`R20260904T171104000000`) and rejects the earlier `R` + `YYDDDHHMMSS` form, so the `DITL_3min` fixtures are renamed to match. In 5.16.0 a manifest lists its files as path strings without checksums (LASP-Libera/libera_utils#106), and `read_all_input_data` reads each entry as a path.
- Rename `Azimuth_FSW_Metadata` to `Azimuth_Motor_Position_FSW_Metadata`. It passes the FSW header azimuth through in every mode; it was 0 in `jpss_only` and -999 with `use_geo: false`.
- `Radiance` and `Azimuth_Motor_Position_FSW_Metadata` declare no `_FillValue`; missing values are NaN.
- Run the per-pixel geometry in Dask tasks of `LIBERA_CAM_GEO_CHUNK_SIZE` frames (default 10), decoupled from the decompression chunk, one `geolocate_frame` call per frame. The kernels are furnished once on the client before the tasks start and a granule no frame of which SPICE covers is rejected; uncovered frames are written as fill and flagged. The worker takes `(T, K)` exposure epochs and a per-pixel epoch index so the two exposures can be geolocated at their own times once known (`LIBSDC-816`); production passes `K = 1`.
- Pin `lasp-curryer` to the `spatial-pixel-geometry` branch (lasp/curryer#188, version 0.5.2) until it is released.

### Fixed

- **Metadata clean up:** Remove the WFOV_HEADER_PARSE_VALID from libera_cam as that field is never useful in L1A files
- Raise a parsed, user-facing `RuntimeError` when the curryer SPICE query fails, and a `ValueError` when any SPICE input the spacecraft fields read (spacecraft ephemeris, attitude, Sun ephemeris) covers no frame of the granule, through curryer's `require_coverage`, instead of writing a fully filled field. Per-frame coverage gaps map onto each variable's `_FillValue`; `use_geo: false` writes the fill values through `create_placeholder_spacecraft_geometry`.
- Write the not-yet-computed per-pixel fields (`Terrain_Corrected_*`) as their product `_FillValue` instead of zeros; the placeholder list in `packaging.py` shrinks as producers land (`LIBSDC-814`).

### Removed

- Remove `calculate_all_pixel_lat_lon_altitude`, `calculate_chunk_geolocation`, the static/dynamic `pixel_mask` options and `prefetch_kernels`; packaging no longer casts or relabels the geolocation variables.

## [0.3.0] - 2026-10-06

### Added

- **Docs**: Document exposure timing equations in `doc/wfov_fsw_header_reference.md`.

### Changed

- **L1A consumption**: Require image-centric WFOV L1A from libera-utils ≥ 5.10.5 (`CAMERA_TIME`, `WFOV_COMPRESSED_IMAGE`). JPEG-LS decompression operates on pre-sliced blob payloads.
- **Exposure metadata**: Convert FPGA actual exposure registers and `WFOV_IMAGE_HEADER_DELTA` to milliseconds; write `Actual_Exposure_Time_1/2` and `Exposure_Delta` in L1B.
- **Azimuth**: Rename to `Azimuth_FSW_Metadata`, drop the `[0, 360]` `valid_range`, and pass through from `WFOV_FSW_HEADER_AZIMUTH_ANGLE`.
- **L1B product definition**: Rename operational-mode fields to `Radiometer_Observation_ID` / `Camera_Observation_ID`; add `Image_Mode` and `Camera_Packet_Index`. The ObsID fields no longer declare `_FillValue`.
- **Dask graph memory**: Decompression tasks carry the L1A file path and row indices instead of embedded JPEG-LS payloads; each worker reads its own rows from the file, so the graph no longer holds every compressed image in memory. In-memory datasets without a file source still embed the arrays.

### Removed

- **L1A consumption**: Remove `reassemble_image_blobs` and the FSW/FPGA header parser.

## [0.2.6] - 2026-08-26

### Added

- **Tuning**: Exposed chunk size configuration via `LIBERA_CAM_CHUNK_SIZE` (default 50) to optimize for specific compute environments. Added operator configuration guide and tuning relationships in [doc/overview.md](doc/overview.md).
- **Dependencies**: Added `netCDF4>=1.6.0` and `distributed>=2026.1.1` to package dependencies for distributed execution and NetCDF file inspection.
- Add `jpss_only` manifest configuration: load only JPSS-SPK and JPSS-CK dynamic kernels, compute per-pixel geolocation using `wfov_pixel_vectors.npy` with `LIBERA_BASE` reference frame (zero-azimuth approximation), and write Azimuth as 0°. Validate required SPICE data products in `read_all_input_data` (production: AZROT-CK + JPSS-SPK + JPSS-CK; jpss_only: JPSS-SPK + JPSS-CK). Reject duplicate kernel types in the manifest. Warn when other SPICE files are listed but skipped. Apply product `_FillValue` for off-Earth and masked pixels.

### Changed

- **Process Parallelism**: Tested with Dask schedulers; **`synchronous`** (default) and **`distributed`** work reliably. **`threads`** and **`processes`** are rejected at runtime because CSPICE/SPICE is not thread-safe within a worker process.
- **Ingestion**: Batch JPEG-LS decompression now builds pre-chunked Dask arrays at ingestion.
- **Geolocation Memory Optimization**: Switched worker geolocation calculations to `float32` preallocated arrays in `libera_cam/geolocation.py`, eliminating duplicate array copies from `np.stack` and reducing worker live memory peak by ~4×.
- Replace `no_geo` manifest key with `use_geo` (default true; `use_geo: false` for ground-calibration placeholder geolocation). Reject incompatible `use_geo: false` + `jpss_only: true` combinations. Align `use_geo: false` placeholder geolocation with product `_FillValue` (-999 lat/lon, -9999 alt) and Azimuth -999.

### Fixed

- Fix Altitude units metadata to meters.

## [0.2.4] - 2026-06-22

### Added

- Add `doc/wfov_fsw_header_reference.md`, documenting the 36-byte FSW metadata block, `img_mode` semantics, and guidance for separating VIDEO double-image pairs that share duplicate `CAMERA_TIME` timestamps. Cross-reference from `metadata_parser.py`.

### Changed

- Set `algorithm_version` dynamically from the installed package at write time (product definition YAML uses `null`, matching `libera_rad`). Added tests to ensure the bundled product definition stays in sync with the repo version.
- Replace `DITL_short` integration fixtures with a smaller `DITL_3min` dataset (~3 minutes of WFOV L1A and SPICE kernels). Add a shared `generate_input_manifest` pytest fixture for manifest-driven integration tests.

## [0.2.3] - 2026-05-28

### Changed

- Use `KernelManager.load_libera_dynamic_kernels(...)` with libera_utils `KernelFileCache` rather than copying `.bc`/`.bsp` kernels into a package-local directory during L1B processing. Dynamic kernels are provided as an explicit **sequence** of sources (e.g. manifest order); `GeolocationKernelConfig` and integration tests pass ordered `.bc`/`.bsp` source lists.

## [0.2.2] - 2026-04-18

### Added

- **Tuning**: Exposed chunk size configuration via `LIBERA_CAM_CHUNK_SIZE` (default 50) to optimize for specific compute environments.

### Changed

- **Production-Ready Architecture**: Transitioned the entire L1B processing pipeline to a fully lazy, memory-efficient execution model using Dask. This is a step towards processing of full-day science products (~3TB uncompressed) on standard compute nodes without OOM errors.
- **Robust Integration**: Unified L1A packet ingestion, radiometric calibration, and SPICE-based geolocation into a coherent, thread-safe pipeline that rigorously adheres to the L1B Product Definition.
- **Lazy Execution**: Implemented strict Dask-based lazy evaluation for the entire pipeline to handle large daily volumes (~3TB/day) within memory limits. Removed eager `.load()` calls during data ingestion.
- **Dask-Optimized Geolocation**: Integrated `libera_cam.geolocation` with Dask `map_blocks` to parallelize SPICE calculations while managing kernel loading safely on workers.
- **Process Parallelism**: Enforced `synchronous` or `processes` scheduling to ensure thread-safety for CSPICE operations.
- **Vectorized Calibration**: Refactored `convert_dn_to_radiance` to use `xr.apply_ufunc` for truly lazy, vectorized operations on Dask arrays.
- **Product Packaging**: Decoupled product formatting logic into `libera_cam.packaging` to enforce strict adherence to L1B Product Definition (renaming, transposing, typing) transparently.
- **Test Refactoring**: Rewrote `tests/unit/test_l1b.py` and `tests/unit/test_camera.py` to decouple them from legacy data files and non-linearity logic. Used rigorous mocking for orchestration tests.
- **Integration Stability**: Updated integration tests to use `synchronous` Dask scheduling to avoid CSPICE kernel conflicts during parallel test execution.

### Removed

- **Cleanup**: Removed unused test data files (`camera_calibration_data.h5`) and obsolete code related to non-linearity corrections.

## [0.2.1] - 2026-04-17

### Added

- Added `add_geolocation_to_dataset` for Dask-based lazy geolocation computation.
- Introduced `GeolocationKernelConfig` to safely configure SPICE kernel managers on Dask workers.
- Added support for both static (2D) and dynamic (3D) pixel masking in geolocation calculations to skip processing of invalid/dark pixels per-timestamp.
- Added explicit `is_dynamic_mask` configuration to replace brittle dimension-based detection.
- Implemented performance assertions in integration benchmarks to detect processing time regressions.
- Added logic to correctly align Time dimensions during Dask `map_blocks` execution when using 3D masks.
- Added `valid_pixel_mask` variable to the L1A dataset to identify valid data pixels (value > 0).

### Changed

- Optimized geolocation memory usage by moving `pointing_vectors` loading to Dask workers via `mmap`, avoiding massive serialization overhead.
- Refined dynamic mask loop to process frames serially within workers to ensure SPICE thread-safety while maintaining chunk-level parallelization.

### Fixed

- Fixed a bug in `calculate_all_pixel_lat_lon_altitude` where static mask results were inconsistently reshaped and assigned.

## [0.2.0] - 2026-04-17

### Added

- Created memory-efficient L1A image parsing in `libera_cam/image_parsing` using Dask for lazy execution.
- Implemented robust L1A packet stitching with a generator-based state machine and validation for offset continuity and SOP/EOP flags.
- Added unit and integration tests for L1A parsing, including handling of corrupted packet streams.
- Integrated `stitching_stats` into final Dataset global attributes for quality reporting.

### Changed

- Refactored `read_l1a_cam_data` to return a lazy Xarray Dataset and utilize mission-wide constants for image dimensions.
- Enhanced image processing diagnostics using `logger.exception` for unexpected failures.
- Improved handling of incomplete images by gracefully discarding corrupted partial blobs instead of crashing.

## [0.1.3] - 2026-04-17

### Added

- Added first draft of geolocation calculations

## [0.1.2] - 2025-12-18

### Added

- Add draft product definition for the L1B camera product to support writing output files during algorithm and pipeline
  testing
