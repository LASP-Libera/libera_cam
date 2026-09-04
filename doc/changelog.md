# Version Changes

## 0.2.8

- Geolocate every pixel of every frame through curryer's `spatial.pixel_geometry` (`geolocate_frame`): `Latitude`, `Longitude`, `Altitude` and the five surface angles (`Solar_Zenith_Surface`, `Viewing_Zenith_Surface`, `Solar_Azimuth_Surface_WRT_North`, `Viewing_Azimuth_Surface_WRT_North`, `Relative_Azimuth_Surface`) are now produced per pixel with the product fills, and the per-pixel `Geolocation_Quality_Flag` (uint16; curryer `SpatialQualityFlags` in bits 0-14, bit 15 = geolocation not run) is added to the product definition. Every pixel that hits the ellipsoid is published; `valid_pixel_mask` no longer gates geolocation. Azimuths that round to 360.0 in float32 are wrapped to 0.
- Run the per-pixel geometry in Dask tasks of `LIBERA_CAM_GEO_CHUNK_SIZE` frames (default 10), decoupled from the decompression chunk, one `geolocate_frame` call per frame. The kernels are furnished once on the client before the tasks start and a granule no frame of which SPICE covers is rejected; uncovered frames are written as fill and flagged. The worker takes `(T, K)` exposure epochs and a per-pixel epoch index so the two exposures can be geolocated at their own times once known (`LIBSDC-816`); production passes `K = 1`.
- Remove `calculate_all_pixel_lat_lon_altitude`, `calculate_chunk_geolocation`, the static/dynamic `pixel_mask` options and `prefetch_kernels`; packaging no longer casts or relabels the geolocation variables.
- Pin `lasp-curryer` to the `spatial-pixel-geometry` branch (lasp/curryer#188, version 0.5.2) until it is released.
- Encode the per-pixel geolocation variables with `shuffle: true`, measured on `DITL_3min`: the granule is 1.6x smaller (225 MB to 144 MB) and the run 1.4x faster (16.3 s to 11.9 s), and every field reads back bit-identical. `Radiance` is pinned `shuffle: false` and `Pixel_Counts` `shuffle: true` because the engines disagree on the default (netCDF4-python shuffles whenever zlib is on, h5py does not), which was costing 16 GB per 12 h granule on `Radiance` under the `netcdf4` engine that AWS runs use. A 12 h daytime granule at 5 s cadence is about 412 GB.
- Record what the write path does under each pairing, measured on scaled fixtures and written up in `doc/overview.md`. `DASK_SCHEDULER=distributed` now works with either engine: the `h5netcdf` + `distributed` failure was `libera_utils` handing `to_netcdf` an open file object, which the scheduler cannot pickle, and `libera_utils` now always passes a path. `h5netcdf` is the better engine on every axis measured (176 s against 217 s and 40.5 against 47.3 MB/frame at 60 frames), and one `netcdf4` + `distributed` run in three failed with a `KeyError` race in which a worker reopened the output before the client's variable definitions were visible to it. No pairing releases blocks as it writes, so memory still grows with granule length; the earlier per-frame memory model is withdrawn because it was measured on the client process alone and understated `distributed` by 4-6x, and `DASK_MEMORY_LIMIT` must be raised above its `8GB` default to write 60 frames under `distributed` at all. Full-length granules remain roadmap item 5.
- Add the per-pixel `Viewing_Azimuth_Surface_WRT_North` and `Solar_Azimuth_Surface_WRT_North` to the product definition, matching the `libera_rad` surface-angle set, so `Relative_Azimuth_Surface` can be derived from them (curryer's CERES convention).
- Write the not-yet-computed per-pixel fields (`Terrain_Corrected_*`) as their product `_FillValue` instead of zeros; the placeholder list in `packaging.py` shrinks as producers land (`LIBSDC-814`).
- Add a SPICE-versus-FSW azimuth diagnostic (`tests/helpers/azimuth_diagnostics.py`, shortest-arc statistics) with an integration test on the DITL data, adapted from the comparison logging in PR #16 (Matt Watwood). Analysis only; the header value is not a production reference.

## 0.2.7

- Populate the spacecraft-level geometry fields from `curryer.compute.geometry.GeometryData`, following the `libera_rad` implementation: `Subsatellite_Latitude`/`_Longitude`/`_Colatitude`, `Subsolar_Latitude`/`_Longitude`/`_Colatitude`, `Radius_of_Satellite_from_Center_of_Earth`, and the granule attribute `Earth_Sun_Distance_AU`, plus the new `Satellite_Position` and `Satellite_Velocity` (J2000, on `EUCLIDEAN_DIM`) and `Satellite_Attitude_Q0..Q3` (ECEF). Fields are requested as `GeometryField` members with a single spacecraft-observer query that serves production and `jpss_only` alike; the observer is validated against the Libera FK (`SPACECRAFT_OBSERVERS`).
- Raise a parsed, user-facing `RuntimeError` when the curryer SPICE query fails or the spacecraft kernels return no coverage for the granule, instead of writing a fully filled granule. Per-frame coverage gaps map onto each variable's `_FillValue`; `use_geo: false` writes the fill values through `create_placeholder_spacecraft_geometry`.
- Derive `Azimuth` from the AZROT-CK as the motor encoder angle (third 1-2-3 Euler angle of `LIBERA_BASE_COORD -> LIBERA_AZ_COORD`, degrees in `[0, 360)`) via `curryer.compute.spatial.frame_to_frame_euler`, matching `libera_rad`. This replaces the FSW image-header `azimuth_angle`, which is recorded in radians and had been written under the degrees product definition unconverted. `jpss_only` writes 0 degrees; `use_geo: false` writes `-999`. Elevation is not reported: the camera is mounted on the azimuth stage and ELSCAN-CK is not a camera L1B input.
- Require `lasp-curryer >= 0.5.0` for the `GeometryField` enum and the SPICE error classifier.

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
