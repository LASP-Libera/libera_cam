"""Tests for camera module"""

# Installed
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

# Local
from libera_cam.camera import convert_dn_to_radiance
from libera_cam.constants import IntegrationTime
from libera_cam.correction_tools.radiometric_corrections import make_synthetic_radiometric_factor


@pytest.mark.parametrize("use_synthetic", [True])
@patch("libera_cam.camera.get_dark_offset")
@patch("libera_cam.camera.get_flat_field_factor")
@patch("libera_cam.camera.get_radiometric_factor")
def test_convert_dn_to_radiance_linear(mock_get_rad, mock_get_flat, mock_get_dark, use_synthetic):
    """
    Test the convert_dn_to_radiance function (Linear Logic).

    Since non-linearity has been removed, the expected equation is:
    Radiance = (DN - DarkOffset) * FlatField * RadiometricFactor
    """
    # 1. Setup Inputs
    # Create a 2x2 image
    pixel_counts = np.array([[100, 200], [300, 400]], dtype=np.uint16)

    # Wrap in DataArray (Time=1, Y=2, X=2)
    dn_da = xr.DataArray(pixel_counts[np.newaxis, ...], dims=("camera_time", "y", "x"), coords={"camera_time": [0]})

    # Integration Time
    int_time = IntegrationTime.SHORT
    int_time_da = xr.DataArray(np.full((1, 2, 2), int_time.value), dims=("camera_time", "y", "x"))

    # 2. Setup Mocks
    # Mock Dark Offset = 0 (scalar or matching shape)
    mock_get_dark.return_value = 0.0

    # Mock Flat Field = 1.0
    mock_get_flat.return_value = 1.0

    # Mock Radiometric Factor (use real calculation for verification)
    rad_factor = make_synthetic_radiometric_factor(int_time)
    mock_get_rad.return_value = rad_factor

    # 3. Run Function
    result_da = convert_dn_to_radiance(dn_da, int_time_da, use_synthetic=use_synthetic)

    # 4. Calculate Expected Result
    # Expected = DN * Factor (since Dark=0, Flat=1)
    expected_radiance = pixel_counts.astype(np.float32) * rad_factor

    # 5. Verify
    # Squeeze time dim for comparison
    np.testing.assert_allclose(
        result_da.values.squeeze(), expected_radiance, rtol=1e-6, err_msg="Linear radiance calculation failed."
    )

    # Verify Metadata
    assert result_da.dtype == np.float32
    assert result_da.shape == (1, 2, 2)

    # Verify Calls
    mock_get_dark.assert_called_once()
    mock_get_flat.assert_called_once()
    mock_get_rad.assert_called_once()
