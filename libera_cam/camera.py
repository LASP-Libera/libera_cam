"""The primary methods used for calculating the l1b camera data"""

import numpy as np
import xarray as xr

from libera_cam.correction_tools import (
    get_dark_offset,
    get_flat_field_factor,
    get_radiometric_factor,
)


def convert_dn_to_radiance(
    dn_images: xr.DataArray,
    int_time_masks: xr.DataArray,
    use_synthetic: bool = False,
    use_exact: bool = False,
):
    """Converts an image of dn (digital number counts) to a radiance image.

    This function applies a series of corrections to the input DN image,including dark offset correction,
    non-linearity correction, flat-fielding correction, and radiometric calibration.

    Parameters
    ----------
    dn_images : xr.DataArray
        A 3D DataArray (time, y, x) representing the input DN images.
    int_time_masks : xr.DataArray
        A 3D DataArray (time, y, x) of integration times used to acquire each pixel.
    use_synthetic : bool
        This is a flag to use synthetic calibration parameters
    use_exact: bool, Optional
        Determines if a known exact calculation factor should be used. Used only if the use_synthetic variable  is also
        True. When this is false, and use_synthetic is true, will return a polynomial estimate of the nonlinearity

    Returns
    -------
    xr.DataArray
        A 3D DataArray representing the radiance images.
        Shape will be the same as `dn_images`.
    """
    if not isinstance(dn_images, xr.DataArray) or not isinstance(int_time_masks, xr.DataArray):
        raise TypeError("Both dn_images and int_time_masks must be Xarray DataArrays.")

    # 1. Dark Offset Correction
    dark_offset = get_dark_offset(int_time_masks, use_synthetic=use_synthetic)
    dark_corrected_counts = dn_images - dark_offset

    # 2. Flat-Fielding Correction
    flat_field_factor = get_flat_field_factor(use_synthetic=use_synthetic)
    flat_field_corrected_counts = dark_corrected_counts * flat_field_factor

    # 3. Radiometric Calibration Coefficient
    radiometric_factor = get_radiometric_factor(int_time_masks, use_synthetic=use_synthetic)
    radiance_image = flat_field_corrected_counts * radiometric_factor

    return radiance_image.astype(np.float32)
