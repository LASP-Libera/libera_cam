"""The commonly used constant values of the camera"""

import importlib.resources as resources
from enum import IntEnum

from cloudpathlib import AnyPath

# EARTH_RADIUS: 6370997.0 # Reference? Can use astropy constants
# Possible ref: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html

PIXEL_COUNT_X = 2048  # Number of pixels in the x direction of the focal plane array (FPA)
PIXEL_COUNT_Y = 2048  # Number of pixels in the y direction of the focal plane array (FPA)
PIXEL_SIZE_X = 0.0055  # Size of the pixel in the x direction of the focal plane array in meters
PIXEL_SIZE_Y = 0.0055  # Size of the pixel in the y direction of the focal plane array in meters
VZA_LIMIT = 63.0  # Maximum viewing zenith angle in degrees
BIT_COUNT = 12  # Number of bits per pixel

# Distance to angle conversion parameters
DISTANCE_TO_ANGLE_COEFFICIENTS = [1.397428e-02, -1.500364e-01, 6.352646e-01, -9.551312e-01, 9.042359, 0]
# Angle to distance conversion parameters
ANGLE_TO_DISTANCE_COEFFICIENTS = [1.21367e-10, 2.60014e-09, -7.50181e-06, -2.84788e-06, 1.17022e-01, 0]
# Percentage of angular bins to sample
ADM_SAMPLE_PERCENT = 100
# Pixel radius of ADM sample spots
ADM_PIXEL_RADIUS = 20

# Calibration coefficients
# See the calculation in radiometric_calibration.py for details on how this value is derived from ground
# calibration data
RADIOMETRIC_SCALING_COEFFICIENT = 1.8737270248520255e-07
LIBERA_CAM_GROUND_CAL_PATH = AnyPath(resources.files("libera_cam").joinpath("data", "ground_calibration"))
GROUND_CAL_PIXEL_MAPPING = LIBERA_CAM_GROUND_CAL_PATH / "wfov_pixel_vectors.npy"

# --- SPICE observer frames ------------------------------------------------------------

# Spacecraft frames the Libera frame kernels define, one per libera_utils kernel set (jpss4,
# noaa20). Libera flies on JPSS-4; NOAA-20 is the alternate bus configuration.
SPACECRAFT_OBSERVERS: tuple[str, ...] = ("JPSS4_SC", "NOAA20_SC")
DEFAULT_SPACECRAFT_OBSERVER = "JPSS4_SC"

# Azimuth motor encoder frames: the CK built from the encoder rotates LIBERA_BASE_COORD into
# LIBERA_AZ_COORD. The WFOV camera is mounted on the azimuth stage (LIBERA_WFOV_CAM_COORD is a
# fixed offset from LIBERA_AZ_COORD), so the azimuth encoder is the only motor in its pointing
# chain; the elevation encoder (LIBERA_AZ_COORD -> LIBERA_EL_COORD) moves the radiometers alone.
AZIMUTH_ENCODER_FRAMES: tuple[str, str] = ("LIBERA_BASE_COORD", "LIBERA_AZ_COORD")


class IntegrationTime(IntEnum):
    """The class defining the length of integration times"""

    # Integration time for short exposures in milliseconds
    SHORT = 1
    # Integration time for long exposures in milliseconds
    LONG = 20


# Default chunk size for the time dimension in Dask arrays.
# This balances memory usage (smaller chunks) against SPICE setup overhead (larger chunks).
# 50 images: ~800 MB raw data + ~2.5 GB float32 geolocation arrays ~= 3.3 GB total per chunk.
DEFAULT_TIME_CHUNK_SIZE = 50
