import h5py
import numpy as np
import pytest

from libera_cam.camera import convert_dn_to_radiance
from libera_cam.constants import IntegrationTime


@pytest.mark.parametrize(("use_exact", "expected_tolerance"), [(True, 1e-5), (False, 1.4)])
def test_synthetic_convert_dn_to_radiance(test_data_path, local_data_path, use_exact, expected_tolerance):
    """Test the convert_dn_to_radiance function using expected data"""
    test_data = h5py.File(test_data_path / "camera_raw_count_example.h5", "r")
    pixel_counts = test_data["cnt_obs0"][:]

    calculated_radiance = convert_dn_to_radiance(
        pixel_counts, IntegrationTime.SHORT, use_synthetic=True, use_exact=use_exact
    )

    reference_answer = h5py.File(test_data_path / "viirs_known_result.h5", "r")
    ref_radiance = reference_answer["rad"][:]

    zero_mask = pixel_counts == 0
    masked_reference = ref_radiance[zero_mask]
    masked_output = calculated_radiance[zero_mask]

    full_mask = masked_reference > 0
    masked_reference = masked_reference[full_mask]
    masked_output = masked_output[full_mask]

    # TODO need to compare these two results, pending how to deal with the random noise of darks
    np.testing.assert_allclose(masked_output, masked_reference, rtol=expected_tolerance)
