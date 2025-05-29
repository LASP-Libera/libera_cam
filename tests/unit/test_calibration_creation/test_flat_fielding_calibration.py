import pytest

from libera_cam.calibration_creation.flat_fielding_calibration import generate_flat_field_factor
from libera_cam.constants import PIXEL_COUNT_X, PIXEL_COUNT_Y


@pytest.mark.parametrize("use_synthetic", [True, False])
def test_generate_flat_field_factor(use_synthetic):
    if use_synthetic:
        scale_factor = generate_flat_field_factor(use_synthetic=True)
        assert scale_factor.shape == (PIXEL_COUNT_X, PIXEL_COUNT_Y)
        assert scale_factor.max() == 1
        assert scale_factor.min() == 1
    else:
        with pytest.raises(NotImplementedError):
            generate_flat_field_factor(use_synthetic=False)
