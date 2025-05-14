"""The module for creating the dark calibrations """

import numpy as np

from libera_cam.constants import BIT_COUNT, PIXEL_COUNT_X, PIXEL_COUNT_Y, IntegrationTime


def generate_dark_offset(
        integration_time: IntegrationTime,
        use_synthetic: bool = False) -> np.ndarray:
    """Calculates and writes the dark offset parameters to a calibration file

    Parameters
    ----------
    integration_time: IntegrationTimes
        The integration time to use for the dark offset
    use_synthetic: bool, Optional
        Determines if synthetic data should be used. Default to False

    Returns
    -------
    dark_offset: np.ndarray
        A 2d array of dark offset values (one value per pixel)
    """
    if use_synthetic:
        dark_offset = make_synthetic_dark_offset(integration_time)
    else:
        # Will be implemented when ground calibration parameter files exist.
        raise NotImplementedError

    return dark_offset


def make_synthetic_dark_offset(integration_time: IntegrationTime) -> np.ndarray:
    """Calculates a synthetic dark offset array.

    Parameters
    ----------
    integration_time: IntegrationTime
        The integration time to use for the dark offset
    camera_config: CameraConfiguration
        Configuration constants for the camera

    Returns
    -------
    dark_offset: np.ndarray
        A 2d array of synthetic dark offset values
    """
    match integration_time:
        case IntegrationTime.SHORT:
            dark_mean = 100.0 / 2.0 ** BIT_COUNT
            dark_std = 50.0 / 2.0 ** BIT_COUNT
        case IntegrationTime.LONG:
            dark_mean = 200.0 / 2.0 ** BIT_COUNT
            dark_std = 100.0 / 2.0 ** BIT_COUNT

    dark_offset = (2.0 ** BIT_COUNT *
                   np.random.normal(
                       loc=dark_mean, scale=dark_std,
                       size=(PIXEL_COUNT_X, PIXEL_COUNT_Y)))

    return dark_offset
