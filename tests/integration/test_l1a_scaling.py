"""Integration tests for L1A fixture scaling helpers."""

import numpy as np
import pytest
import xarray as xr
from libera_utils.l1a.wfov_image_metadata import CAMERA_TIME_COORD
from libera_utils.time import multipart_to_dt64

from tests.helpers.l1a_scaling import multipart_fields_from_dt64

pytestmark = pytest.mark.integration


def test_multipart_time_roundtrip_via_libera_utils():
    """Derived day/ms/us fields must round-trip through libera_utils.multipart_to_dt64."""
    time_data = np.array(["2026-02-12T08:00:01", "2026-02-12T08:00:02"], dtype="datetime64[ns]")
    fields = multipart_fields_from_dt64(time_data)
    roundtrip = multipart_to_dt64(fields, "day", "ms", "us")
    np.testing.assert_array_equal(roundtrip.values, np.array(time_data, dtype="datetime64[ns]"))


def test_make_extended_l1a_doubles_image_count(make_extended_l1a, test_ditl_l1a_file_path, tmp_path):
    output_path = make_extended_l1a(tmp_path / "scaled_l1a.nc", copies=2)

    with xr.open_dataset(output_path) as ds:
        with xr.open_dataset(test_ditl_l1a_file_path) as original:
            assert len(ds[CAMERA_TIME_COORD]) == 2 * len(original[CAMERA_TIME_COORD])
            assert "WFOV_COMPRESSED_IMAGE" in ds


def test_make_extended_l1a_negative_copies_slices_time_range(make_extended_l1a, test_ditl_l1a_file_path, tmp_path):
    output_path = make_extended_l1a(tmp_path / "sliced_l1a.nc", copies=-34)

    with xr.open_dataset(output_path) as ds:
        with xr.open_dataset(test_ditl_l1a_file_path) as original:
            assert len(ds[CAMERA_TIME_COORD]) < len(original[CAMERA_TIME_COORD])
            assert ds[CAMERA_TIME_COORD].values[0] == original[CAMERA_TIME_COORD].values[0]
