"""The module for creating the non-linearity calibrations"""

from libera_cam.correction_tools.non_linearity_corrections import make_synthetic_non_linearity_factor


def generate_non_linearity_factor(measured_pixel_count: int, use_synthetic: bool = False):
    """
    Calculates and returns the non-linearity values to be used for calibration parameters

    Parameters
    ----------
    measured_pixel_count: int
        The measured counts from each pixel
    use_synthetic: bool
        A flag to signal if non-linearity data should be generated from ground calibration or synthetically made
    """
    if use_synthetic:
        scale_factor = make_synthetic_non_linearity_factor(measured_pixel_count)
    else:
        # Will be implemented when the calibration parameter file can be generated.
        raise NotImplementedError

    return scale_factor
