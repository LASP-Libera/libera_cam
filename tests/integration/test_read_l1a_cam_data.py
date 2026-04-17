"""Integration tests for reading L1A camera data from DITL files."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from libera_cam.image_parsing.metadata_parser import extract_dict_from_bytearray
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data, reassemble_image_blobs


@pytest.mark.integration
def test_read_ditl_l1a_cam_data(test_ditl_l1a_file_path):
    l1a_dataset = xr.open_dataset(test_ditl_l1a_file_path)
    image_dataset = read_l1a_cam_data(l1a_dataset)

    assert isinstance(image_dataset.image_data.data, da.Array), "Image data should be a Dask array"
    assert isinstance(image_dataset.integration_mask.data, da.Array), "Integration mask should be a Dask array"

    assert len(image_dataset.camera_time) == 114
    assert image_dataset.image_data.shape == (114, 2048, 2048)
    assert image_dataset.integration_mask.shape == (114, 2048, 2048)

    # count good images using the metadata field "good_image_flag"
    # This triggers computation if good_image_flag depends on image_data
    assert image_dataset.good_image_flag.sum().values == 114

    # Check valid_pixel_mask
    assert "valid_pixel_mask" in image_dataset
    assert isinstance(image_dataset.valid_pixel_mask.data, da.Array), "Valid pixel mask should be a Dask array"

    # Verify logic on a small slice (lazy computation)
    # Check the first time step
    img_slice = image_dataset.image_data.isel(camera_time=0).compute()
    mask_slice = image_dataset.valid_pixel_mask.isel(camera_time=0).compute()

    # Should be True where image > 0
    expected_mask = img_slice > 0
    np.testing.assert_array_equal(mask_slice, expected_mask)


# TODO[LIBSDC-747]: Low priority
@pytest.mark.xfail(reason="Tried and cannot figure out how to validate CRC. Need to work with Beth")
@pytest.mark.integration
def test_crc_check_first_image(test_ditl_l1a_file_path):
    """
    Integration test to verify CRC check on the first image of the DITL file.
    Expected to fail until CRC calculation or data is fixed.
    """
    # Load L1A
    l1a_ds = xr.open_dataset(test_ditl_l1a_file_path)

    # Extract blobs using the helper (eager)
    blobs = list(reassemble_image_blobs(l1a_ds))
    assert len(blobs) > 0, "No blobs found in test file"

    first_blob = blobs[0]

    # Verify CRC
    # extract_dict_from_bytearray runs verify_crc internally if config.validate_crc is True
    # But by default it might just log warning.
    # We explicitly check the 'crc_valid' key returned by extract_dict_from_bytearray

    meta = extract_dict_from_bytearray(first_blob)

    # Check validity
    assert meta["crc_valid"] is True, f"CRC Check failed for first image. Expected CRC: {meta.get('crc')}"
