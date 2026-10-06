"""Integration tests for L1A fixture scaling helpers."""

import pytest
import xarray as xr
from libera_utils.l1a.wfov_image_metadata import CAMERA_TIME_COORD

pytestmark = pytest.mark.integration


def test_make_extended_l1a_doubles_image_count(make_extended_l1a, test_ditl_l1a_file_path, tmp_path):
    output_path = make_extended_l1a(tmp_path / "scaled_l1a.nc", copies=2)

    with xr.open_dataset(output_path) as ds:
        with xr.open_dataset(test_ditl_l1a_file_path) as original:
            assert len(ds[CAMERA_TIME_COORD]) == 2 * len(original[CAMERA_TIME_COORD])
            assert "WFOV_COMPRESSED_IMAGE" in ds


def test_make_extended_l1a_negative_copies_slices_time_range(make_extended_l1a, test_ditl_l1a_file_path, tmp_path):
    output_path = make_extended_l1a(tmp_path / "sliced_l1a.nc", copies=-25)

    with xr.open_dataset(output_path) as ds:
        with xr.open_dataset(test_ditl_l1a_file_path) as original:
            assert len(original[CAMERA_TIME_COORD]) == 3
            assert len(ds[CAMERA_TIME_COORD]) == 2
            assert ds[CAMERA_TIME_COORD].values[0] == original[CAMERA_TIME_COORD].values[0]
