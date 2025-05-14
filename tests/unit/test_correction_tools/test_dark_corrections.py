import numpy as np
import pytest

from libera_cam.constants import IntegrationTime
from libera_cam.correction_tools import get_dark_offset


@pytest.mark.parametrize(
    "use_synthetic",
    [True, False]
)
@pytest.mark.parametrize(
    ("integration_time", "expected_max", "expected_min"),
    [
        (IntegrationTime.SHORT, 441.86, -412.99),
        (IntegrationTime.LONG, 441.86, -412.99),
    ]
)
def test_get_dark_offset(integration_time, use_synthetic,
                         expected_max, expected_min):
    if use_synthetic:
        synth_data = get_dark_offset(integration_time, use_synthetic=use_synthetic)
        assert np.abs(synth_data.max()-expected_max) < 0.01
        assert np.abs(synth_data.min()-expected_min) < 0.01
    else:
        with pytest.raises(NotImplementedError):
            get_dark_offset(integration_time, use_synthetic=False)
