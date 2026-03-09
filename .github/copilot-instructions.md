`libera_cam` is a Python package implementing the Level 1b (L1b) algorithm for the Libera
Wide Field-of-View (WFOV) Camera instrument, developed by the Libera Science Data Center at
LASP. It converts raw DN images to calibrated radiance NetCDF products using a sequential
correction pipeline (dark offset → non-linearity → flat-field → radiometric).

Detailed coding standards, testing conventions, key patterns, and agent restrictions are in
[`.github/instructions/libera_cam.instructions.md`](.github/instructions/libera_cam.instructions.md),
which Copilot applies automatically to all files in this repository.
