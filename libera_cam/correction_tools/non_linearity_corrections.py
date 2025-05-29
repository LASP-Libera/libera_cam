"""The module for finding the most recent non-linearity calibrations to apply as a correction to measurements"""

import sys
from pathlib import Path

# TODO: Change this to netCDF4 after Heather has implemented it and the camera team has confirmed calibration data
import h5py
import numpy as np

from libera_cam.constants import BIT_COUNT, PIXEL_COUNT_X, PIXEL_COUNT_Y
from libera_cam.utils.hdf5_io import load_hdf5_variable, load_hdf5_variable_from_object


def get_non_linearity_factor(
    pixel_counts: np.ndarray, use_synthetic: bool = False, use_exact: bool = False
) -> np.ndarray:
    # TODO: This function will eventually read in calibration parameter files
    """Returns the flat fielding correction from calibration parameters

    Parameters
    ----------
    pixel_counts: np.ndarray
        The pixel counts from the camera
    use_synthetic: bool, Optional
        Determines if synthetic data should be used. Default to False
    use_exact: bool, Optional
        Determines if a known exact calculation factor should be used. Used only if the use_synthetic variable  is also
        True. When this is false, and use_synthetic is true, will return a polynomial estimate of the nonlinearity

    Returns
    -------
    scale_factor: np.ndarray
        A 2d array of flat field corrective values (one value per pixel)
    """
    if use_synthetic:
        if use_exact:
            scale_factor = get_exact_synthetic_non_linearity_factor()
        else:
            scale_factor = make_synthetic_non_linearity_factor(pixel_counts)
    else:
        # Will be implemented when ground calibration parameter files exist.
        raise NotImplementedError

    return scale_factor


def load_non_linearity_parameters(use_synthetic: bool = False) -> np.ndarray:
    """Loads the flat fielding parameters from a file

    Parameters
    ----------
    use_synthetic: bool, Optional
        Determines if synthetic data should be used. Default to False
    """
    data_path = Path(sys.modules[__name__.split(".", maxsplit=1)[0]].__file__).parent / "ground_calibration_data"

    calibration_data = h5py.File(data_path / "camera_calibration_data.h5", "r")

    if use_synthetic:
        calibration_return = load_hdf5_variable_from_object(
            "synthetic_non_linearity_polynomial_coefficients", calibration_data
        )
        calibration_data.close()
        return calibration_return

    raise NotImplementedError


def apply_non_linearity_polynomial(pixel_counts: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Applies the non-linearity polynomial to the pixel counts

    Parameters
    ----------
    pixel_counts: np.ndarray
        The pixel counts from the camera, shape is (2048, 2048)
    coefficients: np.ndarray
        The non-linearity coefficients, shape is (2048, 2048, 6)

    Returns
    -------
    corrected_pixel_counts: np.ndarray
        The pixel counts after the non-linearity correction
    """
    if pixel_counts.shape != (PIXEL_COUNT_X, PIXEL_COUNT_Y):
        raise ValueError(f"Pixel counts must be a 2D array shaped ({PIXEL_COUNT_X}, {PIXEL_COUNT_Y})")
    if coefficients.shape != (PIXEL_COUNT_X, PIXEL_COUNT_Y, 6):
        raise ValueError(f"Coefficients must be a 3D array shaped ({PIXEL_COUNT_X}, {PIXEL_COUNT_Y}, 6)")
    # Coefficients need to be ordered c0,...c5 where the equation is c0 + c1*x ... c5*x^5
    reversed_coefficients = coefficients[..., ::-1]
    # Current order is (row, column, coefficients) needs to be (coefficients, row, column)
    corrected_order_coefficients = reversed_coefficients.transpose((-1, 0, -2))

    corrected_pixel_counts = np.polynomial.polynomial.polyval(pixel_counts, corrected_order_coefficients, tensor=False)
    return corrected_pixel_counts


def make_synthetic_non_linearity_factor(measured_pixel_counts: np.ndarray) -> np.ndarray:
    """A function to create a synthetic non-linearity factor

    Parameters
    ----------
    measured_pixel_counts: np.ndarray
        The measured counts from each pixel
    use_synthetic: bool
        If True, synthetic dark offset data will be used. If False,
        ground calibration data will be used (when available).
        Defaults to False.

    Returns
    -------
    scale_factor: np.ndarray
        The scale factor for the non-linearity
    """
    # Scale the pixel counts to a range of 0 to 1 based on the total bit count
    scaled_response = measured_pixel_counts / (2.0**BIT_COUNT)

    # Load the non-linearity polynomial coefficients
    non_linearity_coefficients = load_non_linearity_parameters(use_synthetic=True)

    # Make non-linearity factor from polynomial coefficients
    adjusted_response = apply_non_linearity_polynomial(scaled_response, non_linearity_coefficients)

    scale_factor = adjusted_response / scaled_response

    return scale_factor


def get_exact_synthetic_non_linearity_factor():
    """Loads the exact non-linearity factor from the test calibration data file"""
    test_calibration_data_path = (
        Path(sys.modules[__name__.split(".", maxsplit=1)[0]].__file__).parent.parent / "tests" / "test_data"
    )
    test_filename = "testing_calibration_data.h5"
    return load_hdf5_variable("reverse_non_linearity_factors", file_path=test_calibration_data_path / test_filename)
