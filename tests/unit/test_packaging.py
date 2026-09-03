"""Tests for the L1B product packaging."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_cam.packaging import package_l1b_product


def _processing_dataset() -> xr.Dataset:
    """Minimal post-processing dataset: internal names, (time, y, x) images, per-frame fields."""
    n_times, n_y, n_x = 2, 3, 4
    times = pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:00:01"])
    return xr.Dataset(
        {
            "image_data": (("camera_time", "y", "x"), da.zeros((n_times, n_y, n_x), dtype=np.uint16)),
            "integration_mask": (("camera_time", "y", "x"), da.zeros((n_times, n_y, n_x), dtype=np.uint8)),
            "Radiance": (("camera_time", "y", "x"), da.zeros((n_times, n_y, n_x), dtype=np.float32)),
            "rad_obs_id": (("camera_time",), np.zeros(n_times, dtype=np.uint16)),
            "cam_obs_id": (("camera_time",), np.zeros(n_times, dtype=np.uint16)),
            "good_image_flag": (("camera_time",), np.zeros(n_times, dtype=np.uint32)),
            # FSW image-header azimuth (radians); not a product variable.
            "azimuth_angle": (("camera_time",), np.array([0.1, 0.2], dtype=np.float32)),
            # Motor azimuth from the SPICE CK (degrees), set during processing.
            "Azimuth": (("camera_time",), np.array([12.5, 13.0], dtype=np.float32)),
            "Satellite_Position": (("camera_time", "EUCLIDEAN_DIM"), np.ones((n_times, 3))),
        },
        coords={"camera_time": times, "y": np.arange(n_y), "x": np.arange(n_x)},
    )


def test_package_transposes_images_and_keeps_state_vector_dimension():
    packaged = package_l1b_product(_processing_dataset())

    assert packaged["Radiance"].dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    assert packaged["Pixel_Counts"].dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    assert packaged["Satellite_Position"].dims == ("CAMERA_TIME", "EUCLIDEAN_DIM")


def test_package_keeps_ck_azimuth_over_header_azimuth():
    """``Azimuth`` is the CK motor angle from processing; the header ``azimuth_angle`` is not renamed onto it."""
    packaged = package_l1b_product(_processing_dataset())

    np.testing.assert_array_equal(packaged["Azimuth"].values, np.array([12.5, 13.0], dtype=np.float32))
    assert packaged["Azimuth"].dtype == np.float32
    assert "azimuth_angle" not in packaged


def test_package_requires_header_azimuth():
    """The header field is dropped deliberately, so a dataset arriving without it is a broken L1A contract."""
    with pytest.raises(ValueError, match="azimuth_angle"):
        package_l1b_product(_processing_dataset().drop_vars("azimuth_angle"))
