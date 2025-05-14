import numpy as np
import pytest

from libera_cam.calibration_creation.radiometric_calibration import generate_radiometric_calibration_factor
from libera_cam.constants import IntegrationTime


@pytest.mark.parametrize(
    "use_synthetic",
    [True, False]
)
@pytest.mark.parametrize(
    ("integration_time", "expected_factor"),
    [
        (IntegrationTime.SHORT, 0.0020833),
        (IntegrationTime.LONG, 0.00010417)
    ]
)
def test_generate_radiometric_factor(use_synthetic, integration_time, expected_factor):
    if use_synthetic:
        rad_factor = generate_radiometric_calibration_factor(integration_time,
                                                             use_synthetic=use_synthetic)
        np.testing.assert_almost_equal(rad_factor, expected_factor, 1e-6)

    else:
        with pytest.raises(NotImplementedError):
            generate_radiometric_calibration_factor(integration_time,
                                        use_synthetic=use_synthetic)
