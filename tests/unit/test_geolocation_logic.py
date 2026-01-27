from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from libera_cam.geolocation import calculate_all_pixel_lat_lon_altitude


@pytest.fixture
def mock_kernel_manager():
    km = MagicMock()
    return km


@pytest.fixture
def mock_pointing_vectors():
    # 4 pixels
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])


@pytest.fixture
def mock_times():
    return pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02"])


@patch("libera_cam.geolocation.PIXEL_COUNT_Y", 2)
@patch("libera_cam.geolocation.PIXEL_COUNT_X", 2)
@patch("libera_cam.geolocation.spatial.compute_ellipsoid_intersection")
@patch("libera_cam.geolocation.spicetime.adapt")
@patch("libera_cam.geolocation.sp.obj.Body")
def test_geolocation_logic_static_path(
    mock_body, mock_adapt, mock_compute, mock_kernel_manager, mock_pointing_vectors, mock_times
):
    """Verify that static mask uses the vectorized path (one call to compute)."""

    # Mock time adaptation
    mock_adapt.return_value = np.array([100.0, 101.0, 102.0])  # Dummy GPS times

    # Mock Body
    mock_body.return_value = MagicMock()

    # Mock return of compute_ellipsoid_intersection
    # Returns tuple (results_df, other1, other2)
    # results_df has lat, lon, alt columns.
    # Input has 3 times * 2 active pixels = 6 results
    n_results = 3 * 2
    mock_results = pd.DataFrame({"lat": np.zeros(n_results), "lon": np.zeros(n_results), "alt": np.zeros(n_results)})
    mock_compute.return_value = (mock_results, None, None)

    # Static mask: [True, False, True, False] -> Indices 0, 2
    static_mask = np.array([True, False, True, False])

    calculate_all_pixel_lat_lon_altitude(
        mock_kernel_manager, mock_times, pointing_vectors=mock_pointing_vectors, pixel_mask=static_mask
    )

    # Assert called ONCE
    assert mock_compute.call_count == 1

    # Check arguments
    # Times should be length 3
    # Vectors should be length 2 (active ones)
    args, kwargs = mock_compute.call_args
    assert len(args[0]) == 3  # Times
    assert len(kwargs["custom_pointing_vectors"]) == 2


@patch("libera_cam.geolocation.PIXEL_COUNT_Y", 2)
@patch("libera_cam.geolocation.PIXEL_COUNT_X", 2)
@patch("libera_cam.geolocation.spatial.compute_ellipsoid_intersection")
@patch("libera_cam.geolocation.spicetime.adapt")
@patch("libera_cam.geolocation.sp.obj.Body")
def test_geolocation_logic_dynamic_path(
    mock_body, mock_adapt, mock_compute, mock_kernel_manager, mock_pointing_vectors, mock_times
):
    """Verify that dynamic mask uses the looped path (N calls to compute)."""

    # Mock time adaptation
    mock_adapt.return_value = np.array([100.0, 101.0, 102.0])  # Dummy GPS times

    # Mock Body
    mock_body.return_value = MagicMock()

    # Dynamic Mask (3 times x 4 pixels)
    # T0: [T, F, F, F] -> 1 pixel
    # T1: [F, T, F, F] -> 1 pixel
    # T2: [F, F, T, T] -> 2 pixels
    dynamic_mask = np.zeros((3, 4), dtype=bool)
    dynamic_mask[0, 0] = True
    dynamic_mask[1, 1] = True
    dynamic_mask[2, 2] = True
    dynamic_mask[2, 3] = True

    # Setup mocks for each call
    # Call 1: 1 result
    res1 = pd.DataFrame({"lat": [0], "lon": [0], "alt": [0]})
    # Call 2: 1 result
    res2 = pd.DataFrame({"lat": [0], "lon": [0], "alt": [0]})
    # Call 3: 2 results
    res3 = pd.DataFrame({"lat": [0, 0], "lon": [0, 0], "alt": [0, 0]})

    mock_compute.side_effect = [(res1, None, None), (res2, None, None), (res3, None, None)]

    calculate_all_pixel_lat_lon_altitude(
        mock_kernel_manager, mock_times, pointing_vectors=mock_pointing_vectors, pixel_mask=dynamic_mask
    )

    # Assert called 3 times (once per timestamp)
    assert mock_compute.call_count == 3

    # Verify calls
    # Call 1 (T0): 1 active vector
    args0, kwargs0 = mock_compute.call_args_list[0]
    assert len(args0[0]) == 1  # 1 time
    assert len(kwargs0["custom_pointing_vectors"]) == 1

    # Call 3 (T2): 2 active vectors
    args2, kwargs2 = mock_compute.call_args_list[2]
    assert len(args2[0]) == 1  # 1 time
    assert len(kwargs2["custom_pointing_vectors"]) == 2


@patch("libera_cam.geolocation.calculate_chunk_geolocation")
@patch("libera_cam.geolocation.prefetch_kernels")
@patch("libera_cam.geolocation.PIXEL_COUNT_Y", 2)
@patch("libera_cam.geolocation.PIXEL_COUNT_X", 2)
def test_add_geolocation_to_dataset_lazy(mock_prefetch, mock_calc_chunk):
    """Verify that add_geolocation_to_dataset returns a lazy dataset with correct graph."""
    import dask.array as da
    import xarray as xr

    from libera_cam.geolocation import (
        GeolocationKernelConfig,
        add_geolocation_to_dataset,
    )

    # Setup lazy dataset
    # Time: 4 steps, chunked by 2
    # Image: (4, 2, 2)
    times = pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02", "2025-01-01T00:00:03"])
    image_data = da.zeros((4, 2, 2), chunks=(2, 2, 2))

    ds = xr.Dataset(
        {"image_data": (("camera_time", "y", "x"), image_data)},
        coords={"camera_time": times, "y": [0, 1], "x": [0, 1]},
    )

    config = GeolocationKernelConfig()

    # Configure mock to return a valid array for metadata inference/computation
    # Dask calls this to determine output array type (numpy vs cupy etc)
    # Output shape should be (Time, Y, X, 3)
    # The chunk size in the test is (2, 2, 2)
    mock_calc_chunk.return_value = np.zeros((2, 2, 2, 3), dtype=np.float64)

    # Call function
    ds_out = add_geolocation_to_dataset(ds, config)

    # 1. Verify Laziness: Mock should NOT be called yet (except possibly for meta inference)
    # Dask map_blocks might call the function once with dummy data to infer meta if not provided
    # but we provided dtype, so it might skip. However, if it calls it, it returns the mock value.

    assert mock_prefetch.call_count == 1
    # assert mock_calc_chunk.call_count == 0  <-- Dask might call it for metadata inference, so we can't strictly assert
    # 0 calls unless we pass meta explicitly in the code.

    # 2. Verify Output Variables
    assert "Latitude" in ds_out
    assert "Longitude" in ds_out
    assert "Altitude" in ds_out

    assert isinstance(ds_out["Latitude"].data, da.Array)
    assert ds_out["Latitude"].dtype == np.float32

    # 3. Verify Graph Structure
    # Compute one chunk

    # Trigger compute of the first chunk of Latitude
    # ds_out["Latitude"] is (Time, Y, X) -> (4, 2, 2)
    # Computed chunk will be (2, 2, 2)
    result_chunk = ds_out["Latitude"][:2, :, :].compute()

    # Now calculate_chunk_geolocation should have been called at least once (inference + compute)
    assert mock_calc_chunk.call_count >= 1
    assert result_chunk.shape == (2, 2, 2)
