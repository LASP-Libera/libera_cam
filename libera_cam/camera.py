"""The primary methods used for calculating the l1b camera data"""
import numpy as np

from libera_cam.constants import IntegrationTime
from libera_cam.correction_tools import (
    get_dark_offset,
    get_flat_field_factor,
    get_non_linearity_factor,
    get_radiometric_factor,
)


def convert_dn_to_radiance(dn_image: np.ndarray,
                           integration_time: IntegrationTime,
                           use_synthetic: bool = False,
                           use_exact: bool = False):
    """Converts an image of dn (digital number counts) to a radiance image.

    This function applies a series of corrections to the input DN image,including dark offset correction,
    non-linearity correction, flat-fielding correction, and radiometric calibration.

    Parameters
    ----------
    dn_image : np.ndarray
        A 2D NumPy array representing the input DN image.
        Shape should be (2048, 2048).
    integration_time : IntegrationTime
        The integration time used to acquire the image.
    use_synthetic : bool
        This is a flag to use synthetic calibration parameters
    use_exact: bool, Optional
        Determines if a known exact calculation factor should be used. Used only if the use_synthetic variable  is also
        True. When this is false, and use_synthetic is true, will return a polynomial estimate of the nonlinearity

    Returns
    -------
    np.ndarray
        A 2D NumPy array representing the radiance image.
        Shape will be the same as `dn_image` (2048, 2048).

    Raises
    ------
    ValueError
        If the input `dn_image` is not a 2D array or does not have the shape (2048, 2048).

    Notes
    -----
    The following corrections are applied in order:
        1. Dark Offset Correction: Removes the dark current signal.
        2. Non-Linearity Correction: Corrects for non-linear response of the sensor.
        3. Flat-Fielding Correction: Corrects for pixel-to-pixel sensitivity variations.
        4. Radiometric Calibration: Converts the corrected counts to radiance units.
    """
    if not isinstance(dn_image, np.ndarray):
        raise TypeError("dn_image must be a NumPy array.")
    if dn_image.ndim != 2:
        raise ValueError("dn_image must be a 2D array.")
    if dn_image.shape != (2048, 2048):
        raise ValueError(f"dn_image must have shape (2048, 2048), but has shape {dn_image.shape}")

    # 1. Dark Offset Correction
    dark_offset = get_dark_offset(integration_time, use_synthetic=use_synthetic)
    dark_corrected_counts = dn_image - dark_offset

    # 2. Non-Linearity Correction
    non_linearity_factor = get_non_linearity_factor(dark_corrected_counts,
                                                    use_synthetic=use_synthetic, use_exact=use_exact)
    non_linearity_corrected_counts = dark_corrected_counts * non_linearity_factor

    # 3. Flat-Fielding Correction
    flat_field_factor = get_flat_field_factor(use_synthetic=use_synthetic)
    flat_field_corrected_counts = non_linearity_corrected_counts * flat_field_factor

    # 4. Radiometric Calibration
    radiometric_factor = get_radiometric_factor(integration_time, use_synthetic=use_synthetic)
    radiance_image = flat_field_corrected_counts * radiometric_factor

    return radiance_image
