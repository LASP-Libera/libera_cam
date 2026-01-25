import h5py
import numpy as np
import pytest
from numpy.ma.testutils import assert_close

from libera_cam.constants import PIXEL_COUNT_X, PIXEL_COUNT_Y
from libera_cam.correction_tools import get_non_linearity_factor
from libera_cam.correction_tools.non_linearity_corrections import (
    apply_non_linearity_polynomial,
    get_exact_synthetic_non_linearity_factor,
    load_non_linearity_parameters,
)
from libera_cam.utils.hdf5_io import load_hdf5_variable


@pytest.mark.parametrize("use_synthetic", [True, False])
def test_get_non_linearity_factor(test_data_path, use_synthetic):
    pixel_counts = load_hdf5_variable("cnt_obs0", test_data_path / "camera_raw_count_example.h5")
    if use_synthetic:
        synth_data = get_non_linearity_factor(pixel_counts, use_synthetic=True)
        assert_close(synth_data.max(), 41.626, decimal=3)
        assert synth_data.shape == (PIXEL_COUNT_X, PIXEL_COUNT_Y)
    else:
        with pytest.raises(NotImplementedError):
            get_non_linearity_factor(pixel_counts, use_synthetic=False)


def test_load_non_linearity_parameters():
    """Test the load_non_linearity_parameters function"""

    # First test the synthetic data
    coefficients = load_non_linearity_parameters(use_synthetic=True)

    assert type(coefficients) is np.ndarray
    assert coefficients.shape == (2048, 2048, 6)
    assert coefficients.dtype == np.float32

    # Test the real data (not implemented yet)
    # TODO[LIBSDC-682] This should likely be its own test, re-eval with new cal data
    with pytest.raises(NotImplementedError):
        load_non_linearity_parameters(use_synthetic=False)


def test_apply_non_linearity_polynomial(test_data_path):
    """Test the apply_non_linearity_polynomial function using expected data"""
    coefficients = load_non_linearity_parameters(use_synthetic=True)

    test_data = h5py.File(test_data_path / "camera_raw_count_example.h5", "r")
    pixel_counts = test_data["cnt_obs0"][:]
    scaled_data = pixel_counts / (2**16)

    returned_values = apply_non_linearity_polynomial(scaled_data, coefficients)

    assert type(returned_values) is np.ndarray
    assert returned_values.shape == (PIXEL_COUNT_X, PIXEL_COUNT_Y)
    assert_close(returned_values.max(), 0.00840, decimal=5)
    assert_close(returned_values.min(), -0.01208, decimal=5)


def test_get_exact_synthetic(test_data_path):
    """Tests that the exact answers are in the test calibration file"""
    exact_data = get_exact_synthetic_non_linearity_factor()
    np.testing.assert_allclose(exact_data, 1.12, atol=0.2)
