"""Unit tests for L1B product packaging."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_cam.packaging import package_l1b_product


def test_package_l1b_product_preserves_surface_geometry_angles():
    """Computed surface geometry angles must not be overwritten with zero placeholders."""
    n_times = 2
    y_size, x_size = 3, 4
    dims = ("camera_time", "y", "x")
    times = pd.date_range("2025-01-01", periods=n_times, freq="s")
    radiance = da.ones((n_times, y_size, x_size), chunks=(2, 3, 4), dtype=np.float32)
    angle_data = da.full((n_times, y_size, x_size), 42.0, chunks=(2, 3, 4), dtype=np.float32)

    ds = xr.Dataset(
        {
            "Radiance": (dims, radiance),
            "Solar_Zenith_Surface": (dims, angle_data),
            "Viewing_Zenith_Surface": (dims, angle_data + 1),
            "Relative_Azimuth_Surface": (dims, angle_data + 2),
            "Latitude": (dims, angle_data),
            "Longitude": (dims, angle_data),
            "Altitude": (dims, angle_data),
            "azimuth_angle": ("camera_time", np.zeros(n_times, dtype=np.float32)),
            "rad_obs_id": ("camera_time", np.zeros(n_times, dtype=np.uint16)),
            "cam_obs_id": ("camera_time", np.zeros(n_times, dtype=np.uint16)),
            "image_data": (dims, da.zeros((n_times, y_size, x_size), chunks=(2, 3, 4), dtype=np.uint16)),
            "integration_mask": (dims, da.zeros((n_times, y_size, x_size), chunks=(2, 3, 4), dtype=np.uint8)),
            "good_image_flag": ("camera_time", np.zeros(n_times, dtype=np.uint32)),
            "Integration_Time": (dims, da.zeros((n_times, y_size, x_size), chunks=(2, 3, 4), dtype=np.uint8)),
        },
        coords={"camera_time": times, "y": range(y_size), "x": range(x_size)},
    )

    packaged = package_l1b_product(ds)

    assert np.all(packaged["Solar_Zenith_Surface"].values == pytest.approx(42.0))
    assert np.all(packaged["Viewing_Zenith_Surface"].values == pytest.approx(43.0))
    assert np.all(packaged["Relative_Azimuth_Surface"].values == pytest.approx(44.0))
