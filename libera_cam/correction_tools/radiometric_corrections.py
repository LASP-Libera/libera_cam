"""The module for finding the most recent radiometric calibrations to apply as a correction to measurements"""

from libera_cam.constants import IntegrationTime


def get_radiometric_factor(integration_time: IntegrationTime, use_synthetic: bool = False) -> float:
    """
    Retrieves the most recent radiometric correction factor for the camera

    Parameters
    ----------
    integration_time: IntegrationTime
        The integration time of the camera in ms
    use_synthetic: bool, Optional
        A flag to signal if radiometric data should be generated from ground calibration or synthetically made
    """
    if use_synthetic:
        scale_factor = make_synthetic_radiometric_factor(integration_time)
    else:
        raise NotImplementedError
    return scale_factor


def make_synthetic_radiometric_factor(
    integration_time: IntegrationTime, band_width: float = 20.0, scaling_coefficient: float = 2.4e4
) -> float:
    """A function to create a synthetic radiometric calibration factor

    Parameters
    ----------
    integration_time: IntegrationTime
        The integration time of the camera in ms
    band_width: float, Optional
        The bandwidth of the camera. Default to 20.0
    scaling_coefficient: float, Optional
        The scaling coefficient for the camera. Default to 2.4e4
        This reflects an email exchange where 0.5 [radiance] approx. 800 DN at 2 ms
    """

    scale_factor = 1.0 / (band_width * scaling_coefficient * (integration_time / 1000.0))

    return scale_factor
