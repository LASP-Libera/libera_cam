"""The module for creating the flat fielding calibrations """
import numpy as np

from libera_cam.correction_tools.flat_field_corrections import make_synthetic_flat_field_factor


def generate_flat_field_factor(use_synthetic: bool = False) -> np.ndarray:
    """Calculates and returns the flat fielding values to be used for calibration parameters

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
        # Will be implemented when ground calibration parameter file can be generated.
        raise NotImplementedError

    return scale_factor
