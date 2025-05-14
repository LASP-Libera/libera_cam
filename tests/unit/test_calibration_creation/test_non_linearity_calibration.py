import h5py
import numpy as np
import pytest

from libera_cam.calibration_creation.non_linearity_calibration import generate_non_linearity_factor
from libera_cam.constants import PIXEL_COUNT_X, PIXEL_COUNT_Y


@pytest.mark.parametrize(
    "use_synthetic",
    [True, False]
)
def test_generate_flat_field_factor(use_synthetic, test_data_path):
    test_data = h5py.File(test_data_path / 'camera_raw_count_example.h5', 'r')
    pixel_counts = test_data['cnt_obs0'][:]

    if use_synthetic:
        nl_factor = generate_non_linearity_factor(pixel_counts, use_synthetic=use_synthetic)
        assert nl_factor.shape == (PIXEL_COUNT_X, PIXEL_COUNT_Y)
        np.testing.assert_almost_equal(nl_factor.max(), 41.62, 0.01)
        assert nl_factor[~np.isinf(nl_factor)].size == 3286769

    else:
        with pytest.raises(NotImplementedError):
            generate_non_linearity_factor(pixel_counts, use_synthetic=use_synthetic)
