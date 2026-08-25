"""Integration tests for reading L1A camera data from DITL files."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data


@pytest.mark.integration
def test_read_ditl_l1a_cam_data(test_ditl_l1a_file_path):
    l1a_dataset = xr.open_dataset(test_ditl_l1a_file_path)
    image_dataset = read_l1a_cam_data(l1a_dataset)

    assert isinstance(image_dataset.image_data.data, da.Array), "Image data should be a Dask array"
    assert isinstance(image_dataset.integration_mask.data, da.Array), "Integration mask should be a Dask array"

    assert len(image_dataset.camera_time) == 3
    assert image_dataset.image_data.shape == (3, 2048, 2048)
    assert image_dataset.integration_mask.shape == (3, 2048, 2048)

    assert image_dataset.good_image_flag.sum().values == 3

    assert "valid_pixel_mask" in image_dataset
    assert isinstance(image_dataset.valid_pixel_mask.data, da.Array), "Valid pixel mask should be a Dask array"

    assert "actual_exposure_time_1" in image_dataset
    assert "actual_exposure_time_2" in image_dataset
    assert "exposure_delta" in image_dataset
    assert "img_mode" in image_dataset
    assert "camera_packet_index" in image_dataset
    assert "rad_obs_id" in image_dataset
    assert "cam_obs_id" in image_dataset

    img_slice = image_dataset.image_data.isel(camera_time=0).compute()
    mask_slice = image_dataset.valid_pixel_mask.isel(camera_time=0).compute()

    expected_mask = img_slice > 0
    np.testing.assert_array_equal(mask_slice, expected_mask)
