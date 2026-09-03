from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from curryer import spicierpy as sp
from curryer.compute import geometry, spatial
from curryer.compute.constants import SpatialQualityFlags as SQF

from libera_cam import geolocation
from libera_cam.config import product_config_path
from libera_cam.geolocation import (
    _FIELD_VARIABLES,
    _GEO_NOT_COMPUTED_FLAG,
    _PIXEL_VARIABLES,
    FrameGeometry,
    GeolocationKernelConfig,
    _require_frame_coverage,
    add_azimuth_to_dataset,
    add_geolocation_to_dataset,
    add_jpss_only_azimuth_to_dataset,
    add_jpss_only_geolocation_to_dataset,
    add_placeholder_azimuth_to_dataset,
    add_placeholder_geolocation_to_dataset,
    add_placeholder_spacecraft_geometry_to_dataset,
    add_spacecraft_geometry_to_dataset,
    calculate_azimuth,
    calculate_chunk_geometry,
    calculate_spacecraft_geometry,
    create_placeholder_spacecraft_geometry,
    granule_earth_sun_distance,
)


@pytest.fixture
def mock_kernel_manager():
    km = MagicMock()
    return km


# --- Per-pixel geometry: worker task and lazy orchestration (geolocate_frame is mocked) ---

FRAME_SHAPE = (3, 5)
GEO_VARIABLES = [variable for variable, _ in _FIELD_VARIABLES.values()]


def _frame(value: float, flag: int) -> FrameGeometry:
    fields = {name: np.full(FRAME_SHAPE, value, dtype=np.float32) for name in _PIXEL_VARIABLES}
    return FrameGeometry(quality_flags=np.full(FRAME_SHAPE, flag, dtype=np.uint16), **fields)


def _frame_dataset(n_times: int, time_chunk: int) -> xr.Dataset:
    times = pd.date_range("2025-01-01", periods=n_times, freq="5s")
    image_data = da.zeros((n_times, *FRAME_SHAPE), chunks=(time_chunk, *FRAME_SHAPE))
    return xr.Dataset(
        {"image_data": (("camera_time", "y", "x"), image_data)},
        coords={"camera_time": times, "y": range(FRAME_SHAPE[0]), "x": range(FRAME_SHAPE[1])},
    )


def _datetimes(n: int) -> np.ndarray:
    return pd.date_range("2025-01-01", periods=n, freq="5s").values


@pytest.fixture
def small_detector():
    with (
        patch("libera_cam.geolocation.PIXEL_COUNT_Y", FRAME_SHAPE[0]),
        patch("libera_cam.geolocation.PIXEL_COUNT_X", FRAME_SHAPE[1]),
    ):
        yield


@pytest.fixture
def worker_mocks(small_detector):
    """The worker's collaborators: kernel manager, pixel-vector file, time conversion, geolocate_frame."""
    with (
        patch("libera_cam.geolocation.KernelManager") as km_cls,
        patch("libera_cam.geolocation.np.load", return_value=np.zeros((*FRAME_SHAPE, 3), dtype=np.float32)),
        patch("libera_cam.geolocation.spicetime.adapt") as adapt,
        patch("libera_cam.geolocation.geolocate_frame") as geolocate,
    ):
        km = km_cls.return_value
        km.__enter__.return_value = km
        yield SimpleNamespace(km_cls=km_cls, km=km, adapt=adapt, geolocate=geolocate)


def test_field_variables_follow_frame_geometry_and_the_product_definition():
    """Blocks are returned in FrameGeometry order; each lands on a per-pixel product variable of its dtype."""
    product_variables = yaml.safe_load(product_config_path.read_text())["variables"]
    assert tuple(_FIELD_VARIABLES) == FrameGeometry._fields
    for name, (variable, dtype) in _FIELD_VARIABLES.items():
        definition = product_variables[variable]
        assert definition["dimensions"] == ["CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y"], name
        assert np.dtype(definition["dtype"]) == np.dtype(dtype), name
    # "Not run" is a bit of its own above curryer's (the YAML side is pinned in test_product_definition).
    assert _GEO_NOT_COMPUTED_FLAG == 0x8000
    assert max(int(f) for f in SQF) < _GEO_NOT_COMPUTED_FLAG


def test_calculate_chunk_geometry_loops_frames_into_typed_blocks(worker_mocks):
    """One geolocate_frame call per frame at its own epoch, results stacked per field with the field's dtype."""
    worker_mocks.adapt.return_value = np.array([1000, 2000])
    worker_mocks.geolocate.side_effect = [_frame(1.0, 0), _frame(2.0, int(SQF.CALC_ELLIPS_NO_INTERSECT))]
    config = GeolocationKernelConfig(dynamic_kernel_sources=["orbit.bsp"])

    outputs = calculate_chunk_geometry(_datetimes(2)[:, None], None, config)

    worker_mocks.km.load_libera_dynamic_kernels.assert_called_once_with(
        ["orbit.bsp"], needs_naif_kernels=True, needs_static_kernels=True
    )
    worker_mocks.km.__exit__.assert_called_once()
    assert len(outputs) == len(FrameGeometry._fields)
    for (name, (_, dtype)), output in zip(_FIELD_VARIABLES.items(), outputs, strict=True):
        assert output.shape == (2, *FRAME_SHAPE), name
        assert output.dtype == dtype, name
    np.testing.assert_array_equal(outputs[0][0], 1.0)
    np.testing.assert_array_equal(outputs[0][1], 2.0)
    np.testing.assert_array_equal(outputs[-1][0], 0)
    np.testing.assert_array_equal(outputs[-1][1], SQF.CALC_ELLIPS_NO_INTERSECT)

    first, second = worker_mocks.geolocate.call_args_list
    np.testing.assert_array_equal(first.args[0], [1000])
    np.testing.assert_array_equal(second.args[0], [2000])
    assert first.args[1] is None
    assert first.args[2] == "LIBERA_WFOV_CAM"
    assert first.args[3].shape == (15, 3)
    assert first.kwargs == {"frame_shape": FRAME_SHAPE}


def test_calculate_chunk_geometry_passes_each_frames_epochs_and_index(worker_mocks):
    """K = 2: every frame gets its own pair of epochs and its own per-pixel index."""
    worker_mocks.adapt.return_value = np.array([10, 11, 20, 21])
    worker_mocks.geolocate.side_effect = [_frame(0.0, 0), _frame(0.0, 0)]
    index = np.zeros((2, *FRAME_SHAPE), dtype=np.uint8)
    index[1, 0, 0] = 1
    exposure_times = np.stack([_datetimes(2), _datetimes(2) + np.timedelta64(25, "ms")], axis=1)

    calculate_chunk_geometry(exposure_times, index, GeolocationKernelConfig(dynamic_kernel_sources=["k"]))

    first, second = worker_mocks.geolocate.call_args_list
    np.testing.assert_array_equal(first.args[0], [10, 11])
    np.testing.assert_array_equal(second.args[0], [20, 21])
    np.testing.assert_array_equal(first.args[1], index[0])
    np.testing.assert_array_equal(second.args[1], index[1])


def test_calculate_chunk_geometry_jpss_only_uses_libera_base(worker_mocks):
    worker_mocks.adapt.return_value = np.array([1000])
    worker_mocks.geolocate.return_value = _frame(0.0, 0)

    calculate_chunk_geometry(
        _datetimes(1)[:, None], None, GeolocationKernelConfig(dynamic_kernel_sources=["k"], jpss_only=True)
    )

    assert worker_mocks.geolocate.call_args.args[2] == "LIBERA_BASE"


@pytest.mark.parametrize(
    ("exposure_times", "exposure_index", "match"),
    [
        (_datetimes(2), None, r"\(T, K\)"),
        (_datetimes(2)[:, None], np.zeros((2, 5, 3), dtype=np.uint8), "exposure_index must be shaped"),
        (_datetimes(2)[:, None], np.zeros((1, 3, 5), dtype=np.uint8), "exposure_index must be shaped"),
    ],
)
def test_calculate_chunk_geometry_rejects_bad_shapes(worker_mocks, exposure_times, exposure_index, match):
    with pytest.raises(ValueError, match=match):
        calculate_chunk_geometry(exposure_times, exposure_index, GeolocationKernelConfig(dynamic_kernel_sources=["k"]))
    worker_mocks.km_cls.assert_not_called()


@pytest.fixture
def coverage_mocks():
    with (
        patch("libera_cam.geolocation.KernelManager") as km_cls,
        patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2, 3])) as adapt,
        patch("libera_cam.geolocation.spatial.pixel_geometry") as pixel_geometry,
    ):
        km = km_cls.return_value
        km.__enter__.return_value = km
        yield SimpleNamespace(km=km, adapt=adapt, pixel_geometry=pixel_geometry)


def _probe(flags: list[int]) -> MagicMock:
    return MagicMock(quality_flags=np.array(flags, dtype=np.int64)[:, None])


def test_require_frame_coverage_probes_the_boresight_once_per_frame(coverage_mocks):
    coverage_mocks.pixel_geometry.return_value = _probe([0, 0, 0])
    config = GeolocationKernelConfig(dynamic_kernel_sources=["orbit.bsp"])

    _require_frame_coverage(config, _datetimes(3), "LIBERA_WFOV_CAM")

    coverage_mocks.km.load_libera_dynamic_kernels.assert_called_once_with(
        ["orbit.bsp"], needs_naif_kernels=True, needs_static_kernels=True
    )
    coverage_mocks.km.__exit__.assert_called_once()
    call = coverage_mocks.pixel_geometry.call_args
    np.testing.assert_array_equal(call.args[0], [1, 2, 3])
    assert call.args[1] == "LIBERA_WFOV_CAM"
    np.testing.assert_array_equal(call.args[2], [[0.0, 0.0, 1.0]])
    assert call.kwargs == {"allow_nans": True}


def test_require_frame_coverage_raises_when_no_frame_is_covered(coverage_mocks):
    gap = int(SQF.SPICE_ERR_MISSING_ATTITUDE | SQF.CALC_ELLIPS_INSUFF_DATA)
    coverage_mocks.pixel_geometry.return_value = _probe([gap, gap, gap])

    with pytest.raises(RuntimeError, match="cover none of the 3 camera frame"):
        _require_frame_coverage(GeolocationKernelConfig(dynamic_kernel_sources=["k"]), _datetimes(3), "LIBERA_WFOV_CAM")


def test_require_frame_coverage_tolerates_and_logs_individual_gaps(coverage_mocks, caplog):
    """A missed pixel is not a coverage gap; an uncovered epoch is, and one covered frame keeps the granule."""
    gap = int(SQF.SPICE_ERR_MISSING_ATTITUDE | SQF.CALC_ELLIPS_INSUFF_DATA)
    coverage_mocks.pixel_geometry.return_value = _probe([gap, 0, int(SQF.CALC_ELLIPS_NO_INTERSECT)])

    with caplog.at_level("WARNING", logger="libera_cam.geolocation"):
        _require_frame_coverage(GeolocationKernelConfig(dynamic_kernel_sources=["k"]), _datetimes(3), "LIBERA_WFOV_CAM")

    assert any("1 of 3 camera frame(s) have no SPICE coverage" in record.message for record in caplog.records)


@pytest.fixture
def orchestration_mocks(small_detector):
    with (
        patch("libera_cam.geolocation._require_frame_coverage") as coverage,
        patch("libera_cam.geolocation.calculate_chunk_geometry") as chunk,
    ):

        def fake_chunk(exposure_times, exposure_index, config):
            assert exposure_times.shape[1] == 1
            assert exposure_index is None
            n = exposure_times.shape[0]
            return tuple(
                np.full((n, *FRAME_SHAPE), i, dtype=dtype) for i, (_, dtype) in enumerate(_FIELD_VARIABLES.values())
            )

        chunk.side_effect = fake_chunk
        yield SimpleNamespace(coverage=coverage, chunk=chunk)


def test_add_geolocation_to_dataset_is_lazy_and_chunked_by_geo_chunk_size(orchestration_mocks, monkeypatch):
    """Tasks of LIBERA_CAM_GEO_CHUNK_SIZE frames, independent of image_data's chunks; nothing runs until compute."""
    monkeypatch.setenv("LIBERA_CAM_GEO_CHUNK_SIZE", "2")
    ds = _frame_dataset(n_times=5, time_chunk=5)
    config = GeolocationKernelConfig(dynamic_kernel_sources=["orbit.bsp"])

    result = add_geolocation_to_dataset(ds, config)

    coverage_call = orchestration_mocks.coverage.call_args
    assert coverage_call.args[0] is config
    np.testing.assert_array_equal(coverage_call.args[1], ds.camera_time.values)
    assert coverage_call.args[2] == "LIBERA_WFOV_CAM"
    assert orchestration_mocks.chunk.call_count == 0
    for variable, dtype in _FIELD_VARIABLES.values():
        assert isinstance(result[variable].data, da.Array), variable
        assert result[variable].dims == ("camera_time", "y", "x")
        assert result[variable].dtype == dtype, variable
        assert result[variable].data.chunks == ((2, 2, 1), (3,), (5,)), variable

    computed = result[GEO_VARIABLES].compute(scheduler="synchronous")

    assert orchestration_mocks.chunk.call_count == 3
    for i, (variable, _) in enumerate(_FIELD_VARIABLES.values()):
        np.testing.assert_array_equal(computed[variable].values, i, err_msg=variable)
    passed_config = orchestration_mocks.chunk.call_args.args[2]
    assert passed_config is config


def test_add_jpss_only_geolocation_switches_the_body(orchestration_mocks):
    ds = _frame_dataset(n_times=2, time_chunk=2)

    result = add_jpss_only_geolocation_to_dataset(ds, GeolocationKernelConfig(dynamic_kernel_sources=["k"]))
    result["Latitude"].compute(scheduler="synchronous")

    assert orchestration_mocks.coverage.call_args.args[2] == "LIBERA_BASE"
    assert orchestration_mocks.chunk.call_args.args[2].jpss_only is True


def test_add_geolocation_requires_kernel_sources_and_camera_time(orchestration_mocks):
    with pytest.raises(ValueError, match="SPICE kernel sources are required"):
        add_geolocation_to_dataset(_frame_dataset(2, 2), GeolocationKernelConfig())
    with pytest.raises(ValueError, match="camera_time"):
        add_geolocation_to_dataset(xr.Dataset(), GeolocationKernelConfig(dynamic_kernel_sources=["k"]))
    orchestration_mocks.coverage.assert_not_called()


@pytest.mark.parametrize("value", ["0", "-3"])
def test_add_geolocation_rejects_geo_chunk_size_below_one(orchestration_mocks, monkeypatch, value):
    monkeypatch.setenv("LIBERA_CAM_GEO_CHUNK_SIZE", value)
    with pytest.raises(ValueError, match="LIBERA_CAM_GEO_CHUNK_SIZE must be >= 1"):
        add_geolocation_to_dataset(_frame_dataset(2, 2), GeolocationKernelConfig(dynamic_kernel_sources=["k"]))
    orchestration_mocks.coverage.assert_not_called()


def test_add_placeholder_geolocation_to_dataset(small_detector):
    """use_geo false: every geolocation variable as its fill, flags marked not-run, chunked like image_data."""
    ds = _frame_dataset(n_times=6, time_chunk=3)

    result = add_placeholder_geolocation_to_dataset(ds)

    for variable, dtype in _FIELD_VARIABLES.values():
        assert isinstance(result[variable].data, da.Array), variable
        assert result[variable].dtype == dtype, variable
        assert result[variable].shape == (6, *FRAME_SHAPE), variable
        assert result[variable].data.chunks[0] == ds["image_data"].data.chunks[0], variable
    for _, variable, fill, _ in _PIXEL_VARIABLES.values():
        np.testing.assert_array_equal(result[variable].values, fill, err_msg=variable)
    np.testing.assert_array_equal(result["Geolocation_Quality_Flag"].values, 0x8000)


def test_add_placeholder_geolocation_requires_dask_image_data(small_detector):
    eager = _frame_dataset(n_times=2, time_chunk=2)
    eager["image_data"] = (("camera_time", "y", "x"), np.zeros((2, *FRAME_SHAPE)))
    with pytest.raises(ValueError, match="Dask-backed 'image_data'"):
        add_placeholder_geolocation_to_dataset(eager)
    with pytest.raises(ValueError, match="Dask-backed 'image_data'"):
        add_placeholder_geolocation_to_dataset(eager.drop_vars("image_data"))


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
