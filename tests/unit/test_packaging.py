"""Tests for the L1B product packaging."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_cam.packaging import _UNIMPLEMENTED_PIXEL_VARIABLES, package_l1b_product


def _processing_dataset() -> xr.Dataset:
    """Minimal post-processing dataset: internal names, (time, y, x) images, per-frame fields."""
    n_times, n_y, n_x = 2, 3, 4
    times = pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:00:01"])
    return xr.Dataset(
        {
            "image_data": (("camera_time", "y", "x"), da.zeros((n_times, n_y, n_x), dtype=np.uint16)),
            "integration_mask": (("camera_time", "y", "x"), da.zeros((n_times, n_y, n_x), dtype=np.uint8)),
            "Radiance": (("camera_time", "y", "x"), da.zeros((n_times, n_y, n_x), dtype=np.float32)),
            "valid_pixel_mask": (
                ("camera_time", "y", "x"),
                da.from_array(np.arange(n_times * n_y * n_x).reshape(n_times, n_y, n_x) % 4 == 0),
            ),
            "l1a_row": (("camera_time",), np.arange(n_times), {"source": "l1a.nc"}),
            "rad_obs_id": (("camera_time",), np.zeros(n_times, dtype=np.uint16)),
            "cam_obs_id": (("camera_time",), np.zeros(n_times, dtype=np.uint16)),
            "good_image_flag": (("camera_time",), np.zeros(n_times, dtype=np.uint32)),
            # FSW image-header azimuth (degrees); packaged as Azimuth_Motor_Position_FSW_Metadata.
            "azimuth_angle": (("camera_time",), np.array([0.1, 0.2], dtype=np.float32)),
            # Motor azimuth from the SPICE CK (degrees), set during processing.
            "Azimuth_Motor_Position_SPICE": (("camera_time",), np.array([12.5, 13.0], dtype=np.float32)),
            "Satellite_Position": (("camera_time", "EUCLIDEAN_DIM"), np.ones((n_times, 3))),
        },
        coords={"camera_time": times, "y": np.arange(n_y), "x": np.arange(n_x)},
    )


def test_package_transposes_images_and_keeps_state_vector_dimension():
    packaged = package_l1b_product(_processing_dataset())

    assert packaged["Radiance"].dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    assert packaged["Pixel_Counts"].dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    assert packaged["Satellite_Position"].dims == ("CAMERA_TIME", "EUCLIDEAN_DIM")


def test_package_keeps_ck_and_header_azimuths_separate():
    """``Azimuth_Motor_Position_SPICE`` is the CK motor angle.

    The header ``azimuth_angle`` becomes ``Azimuth_Motor_Position_FSW_Metadata``.
    """
    packaged = package_l1b_product(_processing_dataset())

    np.testing.assert_array_equal(
        packaged["Azimuth_Motor_Position_SPICE"].values, np.array([12.5, 13.0], dtype=np.float32)
    )
    assert packaged["Azimuth_Motor_Position_SPICE"].dtype == np.float32
    np.testing.assert_array_equal(
        packaged["Azimuth_Motor_Position_FSW_Metadata"].values, np.array([0.1, 0.2], dtype=np.float32)
    )
    assert packaged["Azimuth_Motor_Position_FSW_Metadata"].dtype == np.float32
    assert "azimuth_angle" not in packaged


def test_placeholders_are_nan():
    """Fields without a producer are written as NaN, never as zeros that look like data."""
    packaged = package_l1b_product(_processing_dataset())

    for name in _UNIMPLEMENTED_PIXEL_VARIABLES:
        placeholder = packaged[name]
        assert placeholder.dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
        assert placeholder.dtype == np.float32
        assert placeholder.data.chunks == packaged["Radiance"].data.chunks
        assert np.isnan(placeholder.values).all(), name


def test_camera_mask_is_the_kept_pixel_mask():
    """Camera_Mask carries valid_pixel_mask as 1/0 in product order, and valid_pixel_mask leaves the product."""
    dataset = _processing_dataset()
    kept = dataset["valid_pixel_mask"].values

    packaged = package_l1b_product(dataset)

    assert packaged["Camera_Mask"].dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    assert packaged["Camera_Mask"].dtype == np.uint8
    np.testing.assert_array_equal(packaged["Camera_Mask"].values, kept.transpose(0, 2, 1).astype(np.uint8))
    assert "valid_pixel_mask" not in packaged
    assert "l1a_row" not in packaged


@pytest.mark.parametrize("name", ["valid_pixel_mask", "l1a_row"])
def test_package_requires_the_reader_outputs_it_consumes(name):
    with pytest.raises(ValueError, match=name):
        package_l1b_product(_processing_dataset().drop_vars(name))


def test_packaging_rejects_a_placeholder_variable_that_already_has_data():
    """A producer landing without its name leaving the list must fail, not silently overwrite real data."""
    dataset = _processing_dataset()
    dataset["Terrain_Corrected_Latitude"] = (
        ("camera_time", "y", "x"),
        da.full_like(dataset["Radiance"].data, 42.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="_UNIMPLEMENTED_PIXEL_VARIABLES"):
        package_l1b_product(dataset)


def test_produced_geolocation_variables_pass_through_untouched():
    """Only the terrain fields lack a producer; computed per-pixel geometry is transposed, never overwritten."""
    assert set(_UNIMPLEMENTED_PIXEL_VARIABLES) == {
        "Terrain_Corrected_Latitude",
        "Terrain_Corrected_Longitude",
        "Terrain_Corrected_Altitude",
    }
    dataset = _processing_dataset()
    angles = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    dataset["Solar_Zenith_Surface"] = (("camera_time", "y", "x"), da.from_array(angles))
    dataset["Geolocation_Quality_Flag"] = (("camera_time", "y", "x"), da.full((2, 3, 4), 0x200, dtype=np.uint16))

    packaged = package_l1b_product(dataset)

    assert packaged["Solar_Zenith_Surface"].dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    np.testing.assert_array_equal(packaged["Solar_Zenith_Surface"].values, angles.transpose(0, 2, 1))
    assert packaged["Solar_Zenith_Surface"].attrs == {}
    assert packaged["Geolocation_Quality_Flag"].dtype == np.uint16
    np.testing.assert_array_equal(packaged["Geolocation_Quality_Flag"].values, 0x200)
