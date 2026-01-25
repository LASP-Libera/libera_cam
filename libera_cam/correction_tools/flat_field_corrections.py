"""The module for finding the most recent flat-fielding calibrations to apply as a correction to measurements"""

import numpy as np

from libera_cam.constants import PIXEL_COUNT_X, PIXEL_COUNT_Y


def get_flat_field_factor(use_synthetic: bool = False) -> np.ndarray:
    # TODO[LIBSDC-682]: This function will eventually read in calibration parameter files
    """Returns the flat fielding correction from calibration parameters

    Parameters
    ----------
    use_synthetic: bool, Optional
        Determines if synthetic data should be used. Default to False

    Returns
    -------
    scale_factor: np.ndarray
        A 2d array of flat field corrective values (one value per pixel)
    """
    if use_synthetic:
        scale_factor = make_synthetic_flat_field_factor()
    else:
        # Will be implemented when ground calibration parameter files exist.
        return 1.0  # Placeholder value

    return scale_factor


def make_synthetic_flat_field_factor() -> np.ndarray:
    """Calculates a synthetic flat fielding array.

    Parameters
    ----------
    camera_config: CameraConfiguration
        Configuration constants for the camera

    Returns
    -------
    scale_factor: np.ndarray
        A 2d array of synthetic scaling factors
    """

    # This is an array of ones to not change the values of any pixels.
    scale_factor = np.ones((PIXEL_COUNT_X, PIXEL_COUNT_Y), dtype=np.float32)

    return scale_factor
