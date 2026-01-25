"""The module for finding the most recent dark offset calibrations to apply as a correction to measurements"""

import sys
from pathlib import Path

import numpy as np

from libera_cam.constants import IntegrationTime
from libera_cam.utils.hdf5_io import load_hdf5_variable


def get_dark_offset(integration_time: IntegrationTime, use_synthetic: bool = False) -> np.ndarray:
    """Retrieves the dark offset for a given integration time.

    This function returns the dark offset, which represents the signal
    produced by the camera sensor in the absence of light. This offs
    et
    is dependent on the integration time and is used to correct images
    for this inherent sensor bias.

    Parameters
    ----------
    integration_time : IntegrationTime
        The integration time used during image acquisition, in milliseconds.
    use_synthetic : bool, optional
        If True, synthetic dark offset data will be used. If False,
        ground calibration data will be used (when available).
        Defaults to False.

    Returns
    -------
    np.ndarray
        A 2D NumPy array representing the dark offset. The shape of the
        array is (2048, 2048), corresponding to the camera's pixel array.

    Raises
    ------
    NotImplementedError
        If `use_synthetic` is False and ground calibration data is not
        yet implemented.

    Notes
    -----
    -   The dark offset is a crucial correction for accurate radiometric
        measurements.
    -   When `use_synthetic` is True, the dark offset is read from test_data
        that was generated from random noise.
    -   Future versions will support reading dark offset data from
        ground calibration files when `use_synthetic` is False.
    """
    if use_synthetic:
        test_dark_data_path = (
            Path(sys.modules[__name__.split(".", maxsplit=1)[0]].__file__).parent.parent / "tests" / "test_data"
        )
        test_filename = "testing_calibration_data.h5"
        test_dark_data_path = test_dark_data_path / test_filename
        dark_offset = load_hdf5_variable("dark_pixel_corrections", file_path=test_dark_data_path)
    else:
        # Test data has no offset for now
        dark_offset = 0

    return dark_offset
