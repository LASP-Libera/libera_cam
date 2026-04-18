"""The module for creating the radiometric conversion calibrations"""

import importlib.resources as resources

import pandas as pd

from libera_cam.constants import IntegrationTime
from libera_cam.correction_tools.radiometric_corrections import make_synthetic_radiometric_factor


def generate_radiometric_calibration_factor(
    integration_time: IntegrationTime,
    band_width: float = 20.0,
    scaling_coefficient: float = 2.4e4,
    use_synthetic: bool = False,
) -> float:
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
        scale_factor = make_synthetic_radiometric_factor(
            integration_time, band_width=band_width, scaling_coefficient=scaling_coefficient
        )
    else:
        # See the calculation below for details on how this value is derived
        scale_factor = 1.8737270248520255e-07
    return scale_factor


def calculate_ground_cal_average_radiometric_factor(method: str = "avg") -> float:
    """Calculates the average radiometric calibration factor from the single ground calibration data

    Parameters
    ----------
    method: str, Optional
        The method to use for calculating the average. Default is "avg".
        Currently only "avg" is implemented.

    Returns
    -------
    float
        The average radiometric calibration factor for the specified integration time
    """
    # TODO [LIBSDC-567]: Implement alternative radiometric calibration methods in production version
    if method != "avg":
        raise NotImplementedError("Only 'avg' method is implemented for calculating average radiometric factor.")
    # Use importlib to get the path to the data folder
    data_module = resources.files("libera_cam.data.ground_calibration")
    scaling_factor_file = data_module / "scaling_factor.csv"

    # Load the scaling factors from the CSV file
    scaling_factors = pd.read_csv(scaling_factor_file)
    average_factor = scaling_factors.values.mean()
    return average_factor


if __name__ == "__main__":
    print("Generating Radiometric Calibration Factor from ground calibration:")
    data_module = resources.files("libera_cam.data.ground_calibration")
    scaling_factor_file = data_module / "scaling_factor.csv"
    print(f"Using the average of values in the {scaling_factor_file}")
    factor = calculate_ground_cal_average_radiometric_factor()
    print(f"Average Radiometric Calibration Factor: {factor} [W/m^2/sr/um per DN]")
