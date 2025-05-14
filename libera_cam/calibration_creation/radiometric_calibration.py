"""The module for creating the radiometric conversion calibrations """
from libera_cam.constants import IntegrationTime
from libera_cam.correction_tools.radiometric_corrections import make_synthetic_radiometric_factor


def generate_radiometric_calibration_factor(integration_time: IntegrationTime,
                                            band_width: float = 20.0,
                                            scaling_coefficient: float = 2.4e4,
                                            use_synthetic: bool = False) -> float:
    """Calculates the radiometric calibration factor for the camera

    Parameters
    ----------
    integration_time: IntegrationTime
        The integration time of the camera in ms
    band_width: float, Optional
        The bandwidth of the camera. Default to 20.0
    scaling_coefficient: float, Optional
        The scaling coefficient for the camera. Default to 2.4e4
        This reflects an email exchange where 0.5 [radiance] approx. 800 DN at 2 ms
    use_synthetic: bool, Optional
        A flag to signal if non-linearity data should be generated from ground calibration or synthetically made
    """
    if use_synthetic:
        scale_factor = make_synthetic_radiometric_factor(integration_time,
                                                         band_width=band_width,
                                                         scaling_coefficient=scaling_coefficient)
    else:
        raise NotImplementedError
    return scale_factor
