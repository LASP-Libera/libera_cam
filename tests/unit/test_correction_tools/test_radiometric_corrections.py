import numpy as np
import pytest

from libera_cam.constants import IntegrationTime
from libera_cam.correction_tools import get_radiometric_factor


@pytest.mark.parametrize("use_synthetic", [True, False])
@pytest.mark.parametrize(
    ("integration_time", "expected_factor"), [(IntegrationTime.SHORT, 0.00208), (IntegrationTime.LONG, 0.000104)]
)
def test_get_dark_offset(integration_time, use_synthetic, expected_factor):
    if use_synthetic:
        synth_value = get_radiometric_factor(integration_time, use_synthetic=True)
        assert np.abs(synth_value - expected_factor) < 0.00001
    else:
        with pytest.raises(NotImplementedError):
            get_radiometric_factor(integration_time, use_synthetic=False)
