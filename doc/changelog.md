# Version Changes

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
