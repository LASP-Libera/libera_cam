import numpy as np
import pytest

from libera_cam.calibration_creation.dark_calibration import generate_dark_offset
from libera_cam.constants import PIXEL_COUNT_X, PIXEL_COUNT_Y, IntegrationTime


@pytest.mark.parametrize("use_synthetic", [True, False])
@pytest.mark.parametrize(
    ("integration_time", "expected_mean", "expected_std_dev"),
    [
        (IntegrationTime.SHORT, 100, 50),
        (IntegrationTime.LONG, 200, 100),
    ],
)
def test_generate_dark_offset(integration_time, use_synthetic, expected_mean, expected_std_dev):
    if use_synthetic:
        scale_factor = generate_dark_offset(integration_time, use_synthetic=True)
        assert scale_factor.shape == (PIXEL_COUNT_X, PIXEL_COUNT_Y)
        assert np.abs(scale_factor.mean() - expected_mean) < 1
        assert np.abs(scale_factor.std() - expected_std_dev) < 1
    else:
        with pytest.raises(NotImplementedError):
            generate_dark_offset(integration_time, use_synthetic=False)
