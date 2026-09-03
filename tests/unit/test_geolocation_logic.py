from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from curryer import spicierpy as sp
from curryer.compute import geometry, spatial

from libera_cam import geolocation
from libera_cam.config import product_config_path
from libera_cam.geolocation import (
    GeolocationKernelConfig,
    add_azimuth_to_dataset,
    add_jpss_only_azimuth_to_dataset,
    add_placeholder_azimuth_to_dataset,
    add_placeholder_spacecraft_geometry_to_dataset,
    add_spacecraft_geometry_to_dataset,
    calculate_all_pixel_lat_lon_altitude,
    calculate_azimuth,
    calculate_spacecraft_geometry,
    create_placeholder_spacecraft_geometry,
    granule_earth_sun_distance,
)


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
    mock_calc_chunk.return_value = np.zeros((2, 2, 2, 3), dtype=np.float32)

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


@patch("libera_cam.geolocation.PIXEL_COUNT_Y", 3)
@patch("libera_cam.geolocation.PIXEL_COUNT_X", 5)
def test_add_placeholder_geolocation_to_dataset():
    """Placeholder function adds product fill-value Latitude/Longitude/Altitude."""

    from libera_cam.geolocation import add_placeholder_geolocation_to_dataset

    n_times = 6
    times = pd.date_range("2025-01-01", periods=n_times, freq="s")
    image_data = da.zeros((n_times, 3, 5), chunks=(3, 3, 5))

    ds = xr.Dataset(
        {"image_data": (("camera_time", "y", "x"), image_data)},
        coords={"camera_time": times, "y": range(3), "x": range(5)},
    )

    result = add_placeholder_geolocation_to_dataset(ds)

    for var in ("Latitude", "Longitude", "Altitude"):
        assert var in result, f"{var} missing from result dataset"
        assert isinstance(result[var].data, da.Array), f"{var} should be a dask array"
        assert result[var].dtype == np.float32, f"{var} dtype should be float32"
        assert result[var].shape == (n_times, 3, 5), f"{var} should align with (y, x) dims"
        assert result[var].data.chunks[0] == image_data.chunks[0], f"{var} time chunks should match image_data"

    lat = result["Latitude"].compute().values
    lon = result["Longitude"].compute().values
    alt = result["Altitude"].compute().values
    assert np.all(lat == np.float32(-999))
    assert np.all(lon == np.float32(-999))
    assert np.all(alt == np.float32(-9999))


@patch("libera_cam.geolocation.PIXEL_COUNT_Y", 3)
@patch("libera_cam.geolocation.PIXEL_COUNT_X", 5)
@patch("libera_cam.geolocation.calculate_all_pixel_lat_lon_altitude")
@patch("libera_cam.geolocation.np.load")
@patch("libera_cam.geolocation.KernelManager")
def test_calculate_chunk_geolocation_output_axis_order(mock_km_cls, mock_load, mock_calc_all):
    """Worker output is (T, Y, X, 3) in float32 even when internal calc uses (T, X, Y)."""
    from libera_cam.geolocation import calculate_chunk_geolocation

    mock_km = MagicMock()
    mock_km.__enter__ = MagicMock(return_value=mock_km)
    mock_km.__exit__ = MagicMock(return_value=False)
    mock_km_cls.return_value = mock_km
    mock_load.return_value = np.zeros((15, 3))

    n_times = 2
    lat = np.arange(n_times * 3 * 5, dtype=np.float32).reshape(n_times, 3, 5)
    lon = lat + 100
    alt = lat + 200
    mock_calc_all.return_value = {"latitude": lat, "longitude": lon, "altitude": alt}

    camera_time = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[ns]")
    result = calculate_chunk_geolocation(camera_time, GeolocationKernelConfig())

    assert result.shape == (2, 3, 5, 3)
    assert result.dtype == np.float32
    assert result[0, 2, 1, 0] == lat[0, 2, 1]


@patch("libera_cam.geolocation.PIXEL_COUNT_Y", 2)
@patch("libera_cam.geolocation.PIXEL_COUNT_X", 2)
@patch("libera_cam.geolocation.spatial.compute_ellipsoid_intersection")
@patch("libera_cam.geolocation.spicetime.adapt")
@patch("libera_cam.geolocation.sp.obj.Body")
def test_jpss_only_uses_libera_base_spice_body(
    mock_body, mock_adapt, mock_compute, mock_kernel_manager, mock_pointing_vectors, mock_times
):
    """jpss_only geolocation intersects against LIBERA_BASE instead of LIBERA_WFOV_CAM."""
    mock_adapt.return_value = np.array([100.0, 101.0, 102.0])
    mock_body.return_value = MagicMock()

    n_results = 3 * 2
    mock_results = pd.DataFrame({"lat": np.zeros(n_results), "lon": np.zeros(n_results), "alt": np.zeros(n_results)})
    mock_compute.return_value = (mock_results, None, None)

    static_mask = np.array([True, False, True, False])

    calculate_all_pixel_lat_lon_altitude(
        mock_kernel_manager,
        mock_times,
        pointing_vectors=mock_pointing_vectors,
        pixel_mask=static_mask,
        spice_body="LIBERA_BASE",
    )

    assert mock_compute.call_count == 1
    mock_body.assert_called_with("LIBERA_BASE", frame=True)


# --- Spacecraft-level geometry and motor azimuth (per camera frame, no camera frame involved) ---


@pytest.fixture
def spacecraft_geometry_frame():
    """Two frames of curryer output for every spacecraft field; the second is a SPICE coverage gap."""
    values = {
        "subsatellite_latitude": 10.0,
        "subsatellite_longitude": 20.0,
        "subsatellite_colatitude": 80.0,
        "subsolar_latitude": -5.0,
        "subsolar_longitude": 100.0,
        "subsolar_colatitude": 95.0,
        "spacecraft_radius": 7000.0,
        "earth_sun_distance": 0.985,
        "spacecraft_position_inertial_x": 5000.0,
        "spacecraft_position_inertial_y": 4000.0,
        "spacecraft_position_inertial_z": 1000.0,
        "spacecraft_velocity_inertial_x": -2.9,
        "spacecraft_velocity_inertial_y": 3.5,
        "spacecraft_velocity_inertial_z": 6.0,
        "attitude_q0": 1.0,
        "attitude_q1": 0.0,
        "attitude_q2": 0.0,
        "attitude_q3": 0.0,
    }
    return pd.DataFrame({column: [value, np.nan] for column, value in values.items()})


@pytest.fixture
def camera_dataset():
    times = pd.to_datetime(["2025-01-01T00:00:00", "2025-01-01T00:00:01"])
    return xr.Dataset(coords={"camera_time": times})


@pytest.fixture
def two_timestamps():
    return np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[ns]")


def _all_nan_spacecraft_geometry():
    return pd.DataFrame({column: [np.nan] for field in geolocation._SPACECRAFT_FIELDS for column in field.columns})


def test_spacecraft_field_mapping_matches_curryer_and_product_definition():
    """The variable maps must name curryer columns and reproduce each product variable's fill and dtype."""
    curryer_columns = {column for field in geolocation._SPACECRAFT_FIELDS for column in field.columns}
    product_variables = yaml.safe_load(product_config_path.read_text())["variables"]

    assert set(geolocation._SCALAR_VARIABLES) <= curryer_columns
    assert set(geolocation._VECTOR_VARIABLES) <= set(geolocation._SPACECRAFT_FIELDS)
    for variable, fill, dtype in [*geolocation._SCALAR_VARIABLES.values(), *geolocation._VECTOR_VARIABLES.values()]:
        definition = product_variables[variable]
        assert fill == definition["attributes"]["_FillValue"], variable
        assert np.dtype(dtype) == np.dtype(definition["dtype"]), variable

    azimuth = product_variables[geolocation._AZIMUTH_VARIABLE]
    assert geolocation._FILL_VALUE == azimuth["attributes"]["_FillValue"]
    assert np.dtype(azimuth["dtype"]) == np.float32


@patch("libera_cam.geolocation.geometry.GeometryData")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2]))
def test_calculate_spacecraft_geometry_uses_curryer(
    mock_adapt, mock_geometry_data, mock_kernel_manager, spacecraft_geometry_frame, two_timestamps
):
    """One spacecraft-observer query: every spacecraft field, Earth-fixed attitude, no camera frame."""
    mock_geometry_data.return_value.get_geometry.return_value = spacecraft_geometry_frame

    result = calculate_spacecraft_geometry(mock_kernel_manager, two_timestamps)

    mock_kernel_manager.ensure_known_kernels_are_furnished.assert_called_once()
    mock_geometry_data.assert_called_once_with("JPSS4_SC", attitude_frame=spatial.EARTH_FRAME)
    assert mock_geometry_data.return_value.get_geometry.call_args.kwargs["fields"] == [
        geometry.GeometryField.SUBSATELLITE,
        geometry.GeometryField.SUBSOLAR,
        geometry.GeometryField.SC_RADIUS,
        geometry.GeometryField.EARTH_SUN_DISTANCE,
        geometry.GeometryField.SC_POSITION_INERTIAL,
        geometry.GeometryField.SC_VELOCITY_INERTIAL,
        geometry.GeometryField.SATELLITE_ATTITUDE,
    ]
    assert result is spacecraft_geometry_frame


@pytest.mark.parametrize("observer", ["LIBERA_WFOV_CAM", "LIBERA_BASE"])
def test_calculate_spacecraft_geometry_rejects_unknown_observer(observer, mock_kernel_manager, two_timestamps):
    """A valid SPICE frame used in the wrong role must fail loudly, not compute geometry for the wrong body."""
    with patch("libera_cam.geolocation.geometry.GeometryData") as mock_geometry_data:
        with pytest.raises(ValueError, match="Unsupported spacecraft observer"):
            calculate_spacecraft_geometry(mock_kernel_manager, two_timestamps, spacecraft_observer=observer)
    mock_geometry_data.assert_not_called()
    mock_kernel_manager.ensure_known_kernels_are_furnished.assert_not_called()


@patch("libera_cam.geolocation.spice_error_message", return_value="no coverage for the requested time")
@patch("libera_cam.geolocation.geometry.GeometryData")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1]))
def test_calculate_spacecraft_geometry_raises_friendly_message_on_spice_error(
    mock_adapt, mock_geometry_data, mock_message, mock_kernel_manager, two_timestamps
):
    mock_geometry_data.return_value.get_geometry.side_effect = sp.utils.exceptions.SpiceyError("SPICE(NOFRAMECONNECT)")

    with pytest.raises(RuntimeError, match="no coverage for the requested time"):
        calculate_spacecraft_geometry(mock_kernel_manager, two_timestamps)


@patch("libera_cam.geolocation.geometry.GeometryData")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1]))
def test_calculate_spacecraft_geometry_raises_when_no_coverage(mock_adapt, mock_geometry_data, mock_kernel_manager):
    """All-NaN spacecraft fields mean the kernels miss the granule: a misconfiguration, not a data gap."""
    mock_geometry_data.return_value.get_geometry.return_value = _all_nan_spacecraft_geometry()

    with pytest.raises(RuntimeError, match="no coverage"):
        calculate_spacecraft_geometry(mock_kernel_manager, np.array(["2025-01-01"], dtype="datetime64[ns]"))


@patch("libera_cam.geolocation.geometry.GeometryData")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1]))
def test_calculate_spacecraft_geometry_raises_when_only_sun_fields_covered(
    mock_adapt, mock_geometry_data, mock_kernel_manager
):
    """The Sun-only fields stay finite without spacecraft kernels; coverage is judged on the subsatellite point."""
    frame = _all_nan_spacecraft_geometry()
    for column in (*geometry.GeometryField.SUBSOLAR.columns, *geometry.GeometryField.EARTH_SUN_DISTANCE.columns):
        frame[column] = [1.0]
    mock_geometry_data.return_value.get_geometry.return_value = frame

    with pytest.raises(RuntimeError, match="no coverage"):
        calculate_spacecraft_geometry(mock_kernel_manager, np.array(["2025-01-01"], dtype="datetime64[ns]"))


@patch("libera_cam.geolocation.KernelManager")
@patch("libera_cam.geolocation.geometry.GeometryData")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2]))
def test_add_spacecraft_geometry_maps_columns_and_applies_fill(
    mock_adapt, mock_geometry_data, mock_km_cls, camera_dataset, spacecraft_geometry_frame
):
    """curryer columns map onto product variables; coverage gaps become each variable's _FillValue."""
    mock_geometry_data.return_value.get_geometry.return_value = spacecraft_geometry_frame
    mock_km = mock_km_cls.return_value
    mock_km.__enter__.return_value = mock_km
    config = GeolocationKernelConfig(dynamic_kernel_sources=["some.bsp"])

    result = add_spacecraft_geometry_to_dataset(camera_dataset, config)

    assert result["Subsatellite_Latitude"].dims == ("camera_time",)
    np.testing.assert_allclose(result["Subsatellite_Latitude"].values, [10.0, -999.0])
    np.testing.assert_allclose(result["Subsolar_Longitude"].values, [100.0, -999.0])
    np.testing.assert_allclose(result["Subsatellite_Colatitude"].values, [80.0, -999.0])
    np.testing.assert_allclose(result["Radius_of_Satellite_from_Center_of_Earth"].values, [7000.0, -9999.0])
    assert result["Subsatellite_Latitude"].dtype == np.float32
    assert result["Radius_of_Satellite_from_Center_of_Earth"].dtype == np.float64

    assert result["Satellite_Position"].dims == ("camera_time", "EUCLIDEAN_DIM")
    np.testing.assert_allclose(result["Satellite_Position"].values, [[5000.0, 4000.0, 1000.0], [-9999.0] * 3])
    np.testing.assert_allclose(result["Satellite_Velocity"].values, [[-2.9, 3.5, 6.0], [-999.0] * 3])
    assert result["Satellite_Position"].dtype == np.float64
    assert result["Satellite_Velocity"].dtype == np.float64

    np.testing.assert_allclose(result["Satellite_Attitude_Q0"].values, [1.0, -999.0])
    np.testing.assert_allclose(result["Satellite_Attitude_Q3"].values, [0.0, -999.0])
    assert result["Satellite_Attitude_Q0"].dtype == np.float32

    # The granule attribute ignores the gap rather than averaging it in.
    assert result.attrs["Earth_Sun_Distance_AU"] == pytest.approx(0.985)

    # Kernels are furnished from the config sources and released within the call.
    mock_km.load_libera_dynamic_kernels.assert_called_once_with(
        ["some.bsp"], needs_naif_kernels=True, needs_static_kernels=True
    )
    mock_km.__exit__.assert_called_once()


def test_add_spacecraft_geometry_requires_kernel_sources(camera_dataset):
    with pytest.raises(ValueError, match="SPICE kernel sources are required"):
        add_spacecraft_geometry_to_dataset(camera_dataset, GeolocationKernelConfig())


def test_add_spacecraft_geometry_requires_camera_time():
    with pytest.raises(ValueError, match="camera_time"):
        add_spacecraft_geometry_to_dataset(xr.Dataset(), GeolocationKernelConfig(dynamic_kernel_sources=["k"]))


def test_granule_earth_sun_distance_ignores_gaps():
    assert granule_earth_sun_distance(np.array([np.nan, 0.98, 1.0])) == pytest.approx(0.99)


def test_granule_earth_sun_distance_raises_without_coverage():
    with pytest.raises(RuntimeError, match="Earth-Sun distance has no coverage"):
        granule_earth_sun_distance(np.array([np.nan, np.nan]))


def test_create_placeholder_spacecraft_geometry():
    result = create_placeholder_spacecraft_geometry(5)

    assert len(result) == 5
    expected_columns = [column for field in geolocation._SPACECRAFT_FIELDS for column in field.columns]
    assert list(result.columns) == expected_columns
    assert np.all(result["subsatellite_latitude"].to_numpy() == np.float32(-999))
    assert np.all(result["attitude_q0"].to_numpy() == np.float32(-999))
    assert np.all(result["spacecraft_radius"].to_numpy() == np.float64(-9999))
    assert np.all(result["spacecraft_position_inertial_x"].to_numpy() == np.float64(-9999))
    assert np.all(result["spacecraft_velocity_inertial_x"].to_numpy() == np.float32(-999))


def test_add_placeholder_spacecraft_geometry(camera_dataset):
    result = add_placeholder_spacecraft_geometry_to_dataset(camera_dataset)

    assert np.all(result["Subsatellite_Latitude"].values == np.float32(-999))
    assert np.all(result["Subsolar_Colatitude"].values == np.float32(-999))
    assert np.all(result["Satellite_Attitude_Q2"].values == np.float32(-999))
    assert np.all(result["Radius_of_Satellite_from_Center_of_Earth"].values == np.float64(-9999))
    assert result["Radius_of_Satellite_from_Center_of_Earth"].dims == ("camera_time",)
    assert result["Satellite_Position"].dims == ("camera_time", "EUCLIDEAN_DIM")
    assert np.all(result["Satellite_Position"].values == np.float64(-9999))
    assert np.all(result["Satellite_Velocity"].values == np.float64(-999))
    assert result.attrs["Earth_Sun_Distance_AU"] == -999.0


def _euler_frame(euler3: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"euler1": 0.0, "euler2": 0.0, "euler3": euler3}, index=pd.Index(range(len(euler3)), name="ugps")
    )


@patch("libera_cam.geolocation.spatial.frame_to_frame_euler")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2, 3]))
def test_calculate_azimuth_uses_curryer_frame_euler(mock_adapt, mock_euler, mock_kernel_manager):
    """Azimuth is the third 1-2-3 Euler angle of BASE -> AZ, wrapped to [0, 360); CK gaps become fill."""
    mock_euler.return_value = _euler_frame([-10.0, 350.0, np.nan])
    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02"], dtype="datetime64[ns]")

    azimuth = calculate_azimuth(mock_kernel_manager, timestamps)

    mock_kernel_manager.ensure_known_kernels_are_furnished.assert_called_once()
    call = mock_euler.call_args
    assert call.args[:2] == ("LIBERA_BASE_COORD", "LIBERA_AZ_COORD")
    np.testing.assert_array_equal(call.args[2], [1, 2, 3])
    assert call.kwargs == {"sequence": (1, 2, 3), "allow_nans": True}
    assert azimuth.dtype == np.float32
    np.testing.assert_allclose(azimuth, [350.0, 350.0, -999.0])


@patch("libera_cam.geolocation.spatial.frame_to_frame_euler")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2]))
def test_calculate_azimuth_warns_when_ck_has_no_coverage(mock_adapt, mock_euler, mock_kernel_manager, caplog):
    mock_euler.return_value = _euler_frame([np.nan, np.nan])

    with caplog.at_level("WARNING", logger="libera_cam.geolocation"):
        azimuth = calculate_azimuth(mock_kernel_manager, np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"))

    np.testing.assert_array_equal(azimuth, [-999.0, -999.0])
    assert any("Azimuth CK returned no coverage" in record.message for record in caplog.records)


@patch("libera_cam.geolocation.KernelManager")
@patch("libera_cam.geolocation.spatial.frame_to_frame_euler")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2]))
def test_add_azimuth_to_dataset(mock_adapt, mock_euler, mock_km_cls, camera_dataset):
    mock_euler.return_value = _euler_frame([12.5, np.nan])
    mock_km = mock_km_cls.return_value
    mock_km.__enter__.return_value = mock_km
    config = GeolocationKernelConfig(dynamic_kernel_sources=["azrot.bc"])

    result = add_azimuth_to_dataset(camera_dataset, config)

    assert result["Azimuth"].dims == ("camera_time",)
    assert result["Azimuth"].dtype == np.float32
    np.testing.assert_allclose(result["Azimuth"].values, [12.5, -999.0])
    mock_km.load_libera_dynamic_kernels.assert_called_once_with(
        ["azrot.bc"], needs_naif_kernels=True, needs_static_kernels=True
    )
    mock_km.__exit__.assert_called_once()


def test_add_azimuth_requires_kernel_sources(camera_dataset):
    with pytest.raises(ValueError, match="SPICE kernel sources are required"):
        add_azimuth_to_dataset(camera_dataset, GeolocationKernelConfig())


def test_add_azimuth_requires_camera_time():
    with pytest.raises(ValueError, match="camera_time"):
        add_azimuth_to_dataset(xr.Dataset(), GeolocationKernelConfig(dynamic_kernel_sources=["k"]))


def test_add_placeholder_azimuth(camera_dataset):
    result = add_placeholder_azimuth_to_dataset(camera_dataset)

    assert result["Azimuth"].dims == ("camera_time",)
    assert result["Azimuth"].dtype == np.float32
    assert np.all(result["Azimuth"].values == np.float32(-999))


def test_add_jpss_only_azimuth(camera_dataset):
    result = add_jpss_only_azimuth_to_dataset(camera_dataset)

    assert result["Azimuth"].dtype == np.float32
    assert np.all(result["Azimuth"].values == 0)
