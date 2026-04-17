# Version Changes

## 0.2.0

- Created memory-efficient L1A image parsing in `libera_cam/image_parsing` using Dask for lazy execution.
- Implemented robust L1A packet stitching with a generator-based state machine and validation for offset continuity and SOP/EOP flags.
- Added unit and integration tests for L1A parsing, including handling of corrupted packet streams.
- Added `valid_pixel_mask` variable to the L1A dataset to identify valid data pixels (value > 0).
- Integrated `stitching_stats` into final Dataset global attributes for quality reporting.
- Refactored `read_l1a_cam_data` to return a lazy Xarray Dataset and utilize mission-wide constants for image dimensions.
- Enhanced image processing diagnostics using `logger.exception` for unexpected failures.
- Improved handling of incomplete images by gracefully discarding corrupted partial blobs instead of crashing.

## 0.1.3

- Added first draft of geolocation calculations

## 0.1.2

- Add draft product definition for the L1B camera product to support writing output files during algorithm and pipeline
  testing
