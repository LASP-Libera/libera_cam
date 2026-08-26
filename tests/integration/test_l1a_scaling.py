"""Integration tests for L1A fixture scaling helpers."""

import numpy as np
import pytest
import xarray as xr
from libera_utils.time import multipart_to_dt64

from tests.helpers.l1a_scaling import multipart_fields_from_dt64

pytestmark = pytest.mark.integration


def test_multipart_time_roundtrip_via_libera_utils():
    """Derived day/ms/us fields must round-trip through libera_utils.multipart_to_dt64."""
    time_data = np.array(["2026-02-12T08:00:01", "2026-02-12T08:00:02"], dtype="datetime64[ns]")
    fields = multipart_fields_from_dt64(time_data)
    roundtrip = multipart_to_dt64(fields, "day", "ms", "us")
    np.testing.assert_array_equal(roundtrip.values, np.array(time_data, dtype="datetime64[ns]"))


def test_make_extended_l1a_doubles_packet_count(make_extended_l1a, test_ditl_l1a_file_path, tmp_path):
    output_path = make_extended_l1a(tmp_path / "scaled_l1a.nc", copies=2)

    with xr.open_dataset(output_path) as ds:
        with xr.open_dataset(test_ditl_l1a_file_path) as original:
            packet_dim = original["PACKET_ICIE_TIME"].dims[0]
            assert len(ds[packet_dim]) == 2 * len(original[packet_dim])
            assert len(ds["PACKET_ICIE_TIME"]) == 2 * len(original["PACKET_ICIE_TIME"])


def test_make_extended_l1a_negative_copies_slices_time_range(make_extended_l1a, test_ditl_l1a_file_path, tmp_path):
    output_path = make_extended_l1a(tmp_path / "sliced_l1a.nc", copies=-25)

    with xr.open_dataset(output_path) as ds:
        with xr.open_dataset(test_ditl_l1a_file_path) as original:
            packet_dim = original["PACKET_ICIE_TIME"].dims[0]
            assert len(ds[packet_dim]) < len(original[packet_dim])
            assert ds["PACKET_ICIE_TIME"].values[0] == original["PACKET_ICIE_TIME"].values[0]
