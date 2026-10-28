import logging
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
from dask.utils import key_split

from libera_cam import geolocation
from libera_cam.config import product_config_path
from libera_cam.geolocation import (
    _AZIMUTH_FIELDS,
    _FIELD_VARIABLES,
    _GEO_NOT_COMPUTED_FLAG,
    _PIXEL_FIELDS,
    _PIXEL_VARIABLES,
    FrameGeometry,
    GeolocationKernelConfig,
    add_frame_geometry_to_dataset,
    add_geolocation_to_dataset,
    add_placeholder_frame_geometry_to_dataset,
    add_placeholder_geolocation_to_dataset,
    calculate_azimuth,
    calculate_chunk_geometry,
    calculate_spacecraft_geometry,
    create_placeholder_spacecraft_geometry,
    geolocate_frame,
    granule_earth_sun_distance,
)


@pytest.fixture
def mock_kernel_manager():
    km = MagicMock()
    return km


# --- Per-frame geometry: geolocate_frame (curryer mocked) ---

GRID_SHAPE = (2, 3)
N_PIXELS = 6
EPOCHS = np.array([1_000_000, 2_000_000])
VECTORS = np.tile([0.0, 0.0, 1.0], (N_PIXELS, 1))


def _curryer_result(n_epochs: int) -> dict[str, np.ndarray]:
    """Values encode ``100 * epoch + pixel`` so per-pixel epoch selection is visible; pixel 5 misses.

    Keyed the way curryer keys its return: one entry per requested ``PixelField`` column.
    """
    base = 100.0 * np.arange(n_epochs)[:, None] + np.arange(N_PIXELS)[None, :]
    base[:, 5] = np.nan
    flags = np.zeros((n_epochs, N_PIXELS), dtype=np.int64)
    flags[:, 5] = int(SQF.CALC_ELLIPS_NO_INTERSECT)
    return {
        "longitude": base + 0.1,
        "latitude": base + 0.2,
        "altitude": np.where(np.isnan(base), np.nan, 0.0),
        "solar_zenith": base + 0.3,
        "solar_azimuth": base + 0.4,
        "viewing_zenith": base + 0.5,
        "viewing_azimuth": base + 0.6,
        "relative_azimuth": base + 0.7,
        "quality_flags": flags,
    }


def test_pixel_variable_mapping_is_bound_to_curryer_and_the_product_definition():
    """Every field maps to a real curryer attribute and to a product variable whose dtype it carries."""
    product_variables = yaml.safe_load(product_config_path.read_text())["variables"]
    assert set(_PIXEL_VARIABLES) == set(FrameGeometry._fields) - {"quality_flags"}
    assert set(_AZIMUTH_FIELDS) <= set(_PIXEL_VARIABLES)
    # Every key must be a column curryer actually returns for the fields this product requests.
    requested_columns = {column for field in _PIXEL_FIELDS for column in field.columns}
    assert set(_PIXEL_VARIABLES) | {"quality_flags"} == requested_columns
    for product_name, dtype in _PIXEL_VARIABLES.values():
        definition = product_variables[product_name]
        assert definition["dimensions"] == ["CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y"]
        assert np.dtype(definition["dtype"]) == np.dtype(dtype)


def test_single_epoch_selects_epoch_zero_and_writes_misses_as_nan():
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=_curryer_result(1)) as mock_geometry:
        frame = geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=GRID_SHAPE)

    args = mock_geometry.call_args.args
    np.testing.assert_array_equal(args[0], EPOCHS[:1])
    assert args[1] == "LIBERA_WFOV_CAM"
    np.testing.assert_array_equal(args[2], VECTORS)
    # The declared field list is what reaches curryer, so nothing is computed that this product drops.
    assert mock_geometry.call_args.kwargs["fields"] == list(_PIXEL_FIELDS)

    expected = np.arange(6, dtype=np.float64).reshape(GRID_SHAPE)
    np.testing.assert_allclose(frame.latitude, np.where(expected == 5, np.nan, expected + 0.2), rtol=1e-6)
    np.testing.assert_allclose(frame.longitude, np.where(expected == 5, np.nan, expected + 0.1), rtol=1e-6)
    np.testing.assert_array_equal(frame.altitude, np.where(expected == 5, np.nan, 0.0))
    np.testing.assert_allclose(frame.relative_azimuth, np.where(expected == 5, np.nan, expected + 0.7), rtol=1e-6)
    for name in FrameGeometry._fields:
        values = getattr(frame, name)
        assert values.shape == GRID_SHAPE
        assert values.dtype == (np.uint16 if name == "quality_flags" else np.float32), name
    np.testing.assert_array_equal(frame.quality_flags, np.where(expected == 5, SQF.CALC_ELLIPS_NO_INTERSECT, 0))


def test_two_epochs_select_per_pixel_by_exposure_index():
    exposure_index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=_curryer_result(2)):
        frame = geolocate_frame(EPOCHS, exposure_index, "LIBERA_WFOV_CAM", VECTORS, frame_shape=GRID_SHAPE)

    expected = 100.0 * exposure_index + np.arange(6).reshape(GRID_SHAPE)
    np.testing.assert_allclose(frame.viewing_zenith, np.where(expected % 100 == 5, np.nan, expected + 0.5), rtol=1e-6)
    np.testing.assert_allclose(frame.solar_azimuth[0], [0.4, 101.4, 2.4], rtol=1e-6)
    np.testing.assert_allclose(frame.solar_azimuth[1], [103.4, 4.4, np.nan], rtol=1e-6)


def test_azimuths_just_under_360_wrap_to_zero_after_float32_cast():
    """359.99999 in float64 is 360.0 in float32; curryer's azimuth convention is the half-open [0, 360)."""
    result = _curryer_result(1)
    nearly_full_turn = np.where(np.isnan(result["latitude"]), np.nan, 359.9999999)
    for column in ("solar_azimuth", "viewing_azimuth", "relative_azimuth"):
        result[column] = nearly_full_turn
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result):
        frame = geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=GRID_SHAPE)

    assert np.float32(359.9999999) == np.float32(360.0)
    for name in ("solar_azimuth", "viewing_azimuth", "relative_azimuth"):
        values = getattr(frame, name)
        np.testing.assert_array_equal(values[0], [0.0, 0.0, 0.0], err_msg=name)
        np.testing.assert_array_equal(values[1], [0.0, 0.0, np.nan], err_msg=name)
    # Zeniths are not wrapped.
    assert frame.solar_zenith[0, 0] == np.float32(0.3)


def test_altitude_is_converted_from_kilometers_to_meters():
    result = _curryer_result(1)
    result["altitude"] = np.where(np.isnan(result["altitude"]), np.nan, 1.5)
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result):
        frame = geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=GRID_SHAPE)
    assert frame.altitude[0, 0] == np.float32(1500.0)


@pytest.mark.parametrize(
    ("epochs", "index", "vectors", "match"),
    [
        (np.array([[1, 2]]), None, VECTORS, "non-empty 1-D"),
        (np.array([]), None, VECTORS, "non-empty 1-D"),
        (EPOCHS[:1], None, VECTORS[:4], "pointing_vectors must be shape"),
        (EPOCHS[:1], np.zeros(GRID_SHAPE, dtype=int), VECTORS, "must be None"),
        (EPOCHS, None, VECTORS, "exposure_index is required"),
        (EPOCHS, np.zeros((3, 2), dtype=int), VECTORS, "shaped"),
        (EPOCHS, np.zeros(GRID_SHAPE, dtype=float), VECTORS, "integer array"),
        (EPOCHS, np.full(GRID_SHAPE, 2, dtype=int), VECTORS, "must lie in"),
        (EPOCHS, np.full(GRID_SHAPE, -1, dtype=int), VECTORS, "must lie in"),
    ],
)
def test_geolocate_frame_rejects_bad_inputs(epochs, index, vectors, match):
    with patch("libera_cam.geolocation.spatial.pixel_geometry") as mock_geometry:
        with pytest.raises(ValueError, match=match):
            geolocate_frame(epochs, index, "LIBERA_WFOV_CAM", vectors, frame_shape=GRID_SHAPE)
    mock_geometry.assert_not_called()


@pytest.mark.parametrize("body", ["LIBERA_RAD", "JPSS4_SC", "LIBERA_AZ", ""])
def test_geolocate_frame_rejects_non_camera_bodies(body):
    """A valid SPICE body in the wrong role would geolocate plausibly; only the two camera bodies pass."""
    with patch("libera_cam.geolocation.spatial.pixel_geometry") as mock_geometry:
        with pytest.raises(ValueError, match="camera observer"):
            geolocate_frame(EPOCHS[:1], None, body, VECTORS, frame_shape=GRID_SHAPE)
    mock_geometry.assert_not_called()


def test_jpss_only_body_is_accepted():
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=_curryer_result(1)) as mock_geometry:
        geolocate_frame(EPOCHS[:1], None, "LIBERA_BASE", VECTORS, frame_shape=GRID_SHAPE)
    assert mock_geometry.call_args.args[1] == "LIBERA_BASE"


def test_uncovered_epoch_keeps_its_compound_flag_word():
    """A SPICE gap fills the whole epoch and the compound curryer flag survives the uint16 narrowing."""
    result = _curryer_result(2)
    gap = int(SQF.SPICE_ERR_MISSING_ATTITUDE | SQF.CALC_ELLIPS_INSUFF_DATA)
    result["quality_flags"][1, :] = gap
    # curryer leaves every per-pixel value of an uncovered epoch NaN.
    gap_row = np.array([[False] * 6, [True] * 6])
    for column in _PIXEL_VARIABLES:
        result[column] = np.where(gap_row, np.nan, result[column])
    exposure_index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result):
        frame = geolocate_frame(EPOCHS, exposure_index, "LIBERA_WFOV_CAM", VECTORS, frame_shape=GRID_SHAPE)

    assert gap == 0x104
    np.testing.assert_array_equal(frame.quality_flags[0], [0, gap, 0])
    # An uncovered epoch carries the gap word alone: no intersection was attempted for it.
    np.testing.assert_array_equal(frame.quality_flags[1], [gap, 0, gap])
    np.testing.assert_allclose(frame.latitude[0], [0.2, np.nan, 2.2], rtol=1e-6)
    np.testing.assert_allclose(frame.latitude[1], [np.nan, 4.2, np.nan], rtol=1e-6)
    for name in _PIXEL_VARIABLES:
        assert np.isnan(getattr(frame, name)[exposure_index == 1]).all(), name


@pytest.mark.parametrize("flag", [1 << 15, 1 << 16])
def test_geolocate_frame_rejects_flags_reaching_the_not_computed_bit(flag):
    result = _curryer_result(1)
    result["quality_flags"][0, 0] = flag
    with (
        patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result),
        pytest.raises(RuntimeError, match="bits 0-14"),
    ):
        geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=GRID_SHAPE)


# Pixel ids carried in the vectors' x component, so a fake curryer can tell which pixels it was given.
ID_VECTORS = np.column_stack([np.arange(N_PIXELS, dtype=float), np.zeros(N_PIXELS), np.ones(N_PIXELS)])
KEPT = np.array([[True, False, True], [False, True, True]])


def _curryer_by_pixel_id(epochs, spice_body, vectors, fields, allow_nans):
    """``100 * epoch + pixel id`` for exactly the vectors passed; pixel 5 misses the ellipsoid."""
    ids = vectors[:, 0].astype(int)
    base = 100.0 * np.arange(len(epochs))[:, None] + ids[None, :]
    base[:, ids == 5] = np.nan
    flags = np.where(ids == 5, int(SQF.CALC_ELLIPS_NO_INTERSECT), 0)[None, :].repeat(len(epochs), axis=0)
    columns = {name: base + 0.1 * (i + 1) for i, name in enumerate(_PIXEL_VARIABLES)}
    columns["altitude"] = np.where(np.isnan(base), np.nan, 0.0)
    return columns | {"quality_flags": flags}


def test_pixel_mask_geolocates_only_kept_pixels_and_writes_the_rest_as_nan():
    """curryer sees only the kept pixels' vectors; dropped pixels are NaN with bit 15 alone."""
    with patch("libera_cam.geolocation.spatial.pixel_geometry", side_effect=_curryer_by_pixel_id) as mock_geometry:
        masked = geolocate_frame(
            EPOCHS[:1], None, "LIBERA_WFOV_CAM", ID_VECTORS, frame_shape=GRID_SHAPE, pixel_mask=KEPT
        )
        full = geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", ID_VECTORS, frame_shape=GRID_SHAPE)

    np.testing.assert_array_equal(mock_geometry.call_args_list[0].args[2], ID_VECTORS[KEPT.ravel()])
    for name in _PIXEL_VARIABLES:
        values = getattr(masked, name)
        assert values.shape == GRID_SHAPE, name
        assert values.dtype == np.float32, name
        np.testing.assert_array_equal(values[KEPT], getattr(full, name)[KEPT], err_msg=name)
        assert np.isnan(values[~KEPT]).all(), name
    np.testing.assert_array_equal(masked.quality_flags[KEPT], full.quality_flags[KEPT])
    np.testing.assert_array_equal(masked.quality_flags[~KEPT], _GEO_NOT_COMPUTED_FLAG)
    # Pixel 5 is kept and misses the ellipsoid: curryer's flag, not "not computed".
    assert masked.quality_flags[1, 2] == SQF.CALC_ELLIPS_NO_INTERSECT


def test_pixel_mask_keeps_each_kept_pixels_epoch_selection():
    exposure_index = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    with patch("libera_cam.geolocation.spatial.pixel_geometry", side_effect=_curryer_by_pixel_id):
        masked = geolocate_frame(
            EPOCHS, exposure_index, "LIBERA_WFOV_CAM", ID_VECTORS, frame_shape=GRID_SHAPE, pixel_mask=KEPT
        )
        full = geolocate_frame(EPOCHS, exposure_index, "LIBERA_WFOV_CAM", ID_VECTORS, frame_shape=GRID_SHAPE)

    np.testing.assert_array_equal(masked.latitude[KEPT], full.latitude[KEPT])
    np.testing.assert_allclose(masked.latitude[0], [0.1, np.nan, 102.1], rtol=1e-6)


def test_pixel_mask_with_no_kept_pixels_skips_curryer():
    with patch("libera_cam.geolocation.spatial.pixel_geometry") as mock_geometry:
        frame = geolocate_frame(
            EPOCHS[:1],
            None,
            "LIBERA_WFOV_CAM",
            ID_VECTORS,
            frame_shape=GRID_SHAPE,
            pixel_mask=np.zeros(GRID_SHAPE, bool),
        )
    mock_geometry.assert_not_called()
    assert np.isnan(frame.latitude).all()
    np.testing.assert_array_equal(frame.quality_flags, _GEO_NOT_COMPUTED_FLAG)


@pytest.mark.parametrize("mask", [np.ones((3, 2), dtype=bool), np.ones(GRID_SHAPE, dtype=np.uint8)])
def test_geolocate_frame_rejects_a_mask_that_is_not_boolean_and_frame_shaped(mask):
    with patch("libera_cam.geolocation.spatial.pixel_geometry") as mock_geometry:
        with pytest.raises(ValueError, match="pixel_mask must be a boolean array"):
            geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", ID_VECTORS, frame_shape=GRID_SHAPE, pixel_mask=mask)
    mock_geometry.assert_not_called()


# --- Per-pixel geometry: worker task and lazy orchestration (geolocate_frame is mocked) ---

FRAME_SHAPE = (3, 5)
GEO_VARIABLES = [variable for variable, _ in _FIELD_VARIABLES.values()]


def _frame(value: float, flag: int) -> FrameGeometry:
    fields = {name: np.full(FRAME_SHAPE, value, dtype=np.float32) for name in _PIXEL_VARIABLES}
    return FrameGeometry(quality_flags=np.full(FRAME_SHAPE, flag, dtype=np.uint16), **fields)


L1A_SOURCE = "s3://l1a/LIBERA_L1A_WFOV.nc"


def _kept_pixels(n_times: int) -> np.ndarray:
    """A different kept-pixel pattern per frame, so a task handed another frame's mask shows."""
    return (np.arange(n_times)[:, None, None] + np.arange(np.prod(FRAME_SHAPE)).reshape(FRAME_SHAPE)) % 3 == 0


def _frame_dataset(n_times: int, time_chunk: int) -> xr.Dataset:
    times = pd.date_range("2025-01-01", periods=n_times, freq="5s")
    image_data = da.zeros((n_times, *FRAME_SHAPE), chunks=(time_chunk, *FRAME_SHAPE))
    return xr.Dataset(
        {
            "image_data": (("camera_time", "y", "x"), image_data),
            # L1A rows distinct from the frame positions, so a task handed the wrong rows shows.
            "l1a_row": (("camera_time",), 2 * np.arange(n_times) + 1, {"source": L1A_SOURCE}),
        },
        coords={"camera_time": times, "y": range(FRAME_SHAPE[0]), "x": range(FRAME_SHAPE[1])},
        # Set by add_frame_geometry_to_dataset, which per-pixel geolocation requires first.
        attrs={"Earth_Sun_Distance_AU": 1.0},
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

    mask = _kept_pixels(2)
    outputs = calculate_chunk_geometry(_datetimes(2)[:, None], None, mask, config)

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
    assert set(first.kwargs) == {"frame_shape", "pixel_mask"}
    assert first.kwargs["frame_shape"] == FRAME_SHAPE
    np.testing.assert_array_equal(first.kwargs["pixel_mask"], mask[0])
    np.testing.assert_array_equal(second.kwargs["pixel_mask"], mask[1])


def test_calculate_chunk_geometry_passes_each_frames_epochs_and_index(worker_mocks):
    """K = 2: every frame gets its own pair of epochs and its own per-pixel index."""
    worker_mocks.adapt.return_value = np.array([10, 11, 20, 21])
    worker_mocks.geolocate.side_effect = [_frame(0.0, 0), _frame(0.0, 0)]
    index = np.zeros((2, *FRAME_SHAPE), dtype=np.uint8)
    index[1, 0, 0] = 1
    exposure_times = np.stack([_datetimes(2), _datetimes(2) + np.timedelta64(25, "ms")], axis=1)

    calculate_chunk_geometry(
        exposure_times, index, _kept_pixels(2), GeolocationKernelConfig(dynamic_kernel_sources=["k"])
    )

    first, second = worker_mocks.geolocate.call_args_list
    np.testing.assert_array_equal(first.args[0], [10, 11])
    np.testing.assert_array_equal(second.args[0], [20, 21])
    np.testing.assert_array_equal(first.args[1], index[0])
    np.testing.assert_array_equal(second.args[1], index[1])


def test_calculate_chunk_geometry_jpss_only_uses_libera_base(worker_mocks):
    worker_mocks.adapt.return_value = np.array([1000])
    worker_mocks.geolocate.return_value = _frame(0.0, 0)

    calculate_chunk_geometry(
        _datetimes(1)[:, None],
        None,
        _kept_pixels(1),
        GeolocationKernelConfig(dynamic_kernel_sources=["k"], jpss_only=True),
    )

    assert worker_mocks.geolocate.call_args.args[2] == "LIBERA_BASE"


@pytest.mark.parametrize(
    ("exposure_times", "exposure_index", "pixel_mask", "match"),
    [
        (_datetimes(2), None, _kept_pixels(2), r"\(T, K\)"),
        (_datetimes(2)[:, None], np.zeros((2, 5, 3), dtype=np.uint8), _kept_pixels(2), "exposure_index must be shaped"),
        (_datetimes(2)[:, None], np.zeros((1, 3, 5), dtype=np.uint8), _kept_pixels(2), "exposure_index must be shaped"),
        (_datetimes(2)[:, None], None, _kept_pixels(1), "pixel_mask must be shaped"),
        (_datetimes(2)[:, None], None, np.ones((2, 5, 3), dtype=bool), "pixel_mask must be shaped"),
    ],
)
def test_calculate_chunk_geometry_rejects_bad_shapes(worker_mocks, exposure_times, exposure_index, pixel_mask, match):
    with pytest.raises(ValueError, match=match):
        calculate_chunk_geometry(
            exposure_times, exposure_index, pixel_mask, GeolocationKernelConfig(dynamic_kernel_sources=["k"])
        )
    worker_mocks.km_cls.assert_not_called()


@pytest.fixture
def orchestration_mocks(small_detector):
    def fake_kept_pixels(source, rows):
        assert source == L1A_SOURCE
        return _kept_pixels(max(rows) + 1)[rows]  # the pattern follows the L1A row, not the position

    with (
        patch("libera_cam.geolocation.calculate_chunk_geometry") as chunk,
        patch("libera_cam.geolocation.kept_pixels_from_source", side_effect=fake_kept_pixels),
    ):

        def fake_chunk(exposure_times, exposure_index, pixel_mask, config):
            assert exposure_times.shape[1] == 1
            assert exposure_index is None
            n = exposure_times.shape[0]
            assert pixel_mask.shape == (n, *FRAME_SHAPE)
            assert pixel_mask.dtype == np.bool_
            return tuple(
                np.full((n, *FRAME_SHAPE), i, dtype=dtype) for i, (_, dtype) in enumerate(_FIELD_VARIABLES.values())
            )

        chunk.side_effect = fake_chunk
        yield SimpleNamespace(chunk=chunk)


@pytest.mark.parametrize("time_chunk", [5, 3])
def test_add_geolocation_to_dataset_is_lazy_and_chunked_by_geo_chunk_size(orchestration_mocks, monkeypatch, time_chunk):
    """Tasks of LIBERA_CAM_GEO_CHUNK_SIZE frames, independent of image_data's chunks; nothing runs until compute.

    A decompression chunk of 3 against a geometry chunk of 2 puts frames 2-3 in one task drawn from two
    source blocks, which must still receive exactly those frames' mask.
    """
    monkeypatch.setenv("LIBERA_CAM_GEO_CHUNK_SIZE", "2")
    ds = _frame_dataset(n_times=5, time_chunk=time_chunk)
    config = GeolocationKernelConfig(dynamic_kernel_sources=["orbit.bsp"])

    result = add_geolocation_to_dataset(ds, config)

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
    passed_config = orchestration_mocks.chunk.call_args.args[3]
    assert passed_config is config
    # Each geometry task takes no Dask inputs, so distributed holds it back until a worker thread is
    # free of computing and compressing earlier blocks.
    dependencies = result["Latitude"].data.__dask_graph__().get_all_dependencies()
    geometry_keys = [key for key in dependencies if key_split(key) == "calculate_chunk_geometry_from_l1a"]
    assert len(geometry_keys) == 3
    assert all(dependencies[key] == set() for key in geometry_keys)
    # Each task receives the mask of exactly the frames whose epochs it receives.
    row_of = dict(zip(ds.camera_time.values, ds.l1a_row.values, strict=True))
    for call in orchestration_mocks.chunk.call_args_list:
        rows = [row_of[time] for time in call.args[0][:, 0]]
        np.testing.assert_array_equal(call.args[2], _kept_pixels(max(rows) + 1)[rows])


@pytest.mark.parametrize("earth_sun_distance", [None, np.nan])
def test_add_geolocation_requires_the_per_frame_geometry_first(orchestration_mocks, earth_sun_distance):
    """Without add_frame_geometry_to_dataset's kernel load and coverage check, nothing is scheduled."""
    ds = _frame_dataset(2, 2)
    if earth_sun_distance is None:
        del ds.attrs["Earth_Sun_Distance_AU"]
    else:
        ds.attrs["Earth_Sun_Distance_AU"] = earth_sun_distance
    with pytest.raises(ValueError, match="add_frame_geometry_to_dataset"):
        add_geolocation_to_dataset(ds, GeolocationKernelConfig(dynamic_kernel_sources=["k"]))
    assert "Latitude" not in ds


def test_add_geolocation_requires_kernel_sources_and_camera_time(orchestration_mocks):
    with pytest.raises(ValueError, match="SPICE kernel sources are required"):
        add_geolocation_to_dataset(_frame_dataset(2, 2), GeolocationKernelConfig())
    with pytest.raises(ValueError, match="camera_time"):
        add_geolocation_to_dataset(xr.Dataset(), GeolocationKernelConfig(dynamic_kernel_sources=["k"]))


@pytest.mark.parametrize("breakage", ["missing", "float rows", "no source"])
def test_add_geolocation_requires_the_l1a_rows(orchestration_mocks, breakage):
    """Without each frame's L1A row and file there is no way to rebuild the kept-pixel masks."""
    ds = _frame_dataset(2, 2)
    if breakage == "missing":
        ds = ds.drop_vars("l1a_row")
    elif breakage == "float rows":
        ds["l1a_row"] = ds["l1a_row"].astype(float)
    else:
        ds["l1a_row"].attrs = {}
    with pytest.raises(ValueError, match="'l1a_row'"):
        add_geolocation_to_dataset(ds, GeolocationKernelConfig(dynamic_kernel_sources=["k"]))


@pytest.mark.parametrize("value", ["0", "-3"])
def test_add_geolocation_rejects_geo_chunk_size_below_one(orchestration_mocks, monkeypatch, value):
    monkeypatch.setenv("LIBERA_CAM_GEO_CHUNK_SIZE", value)
    with pytest.raises(ValueError, match="LIBERA_CAM_GEO_CHUNK_SIZE must be >= 1"):
        add_geolocation_to_dataset(_frame_dataset(2, 2), GeolocationKernelConfig(dynamic_kernel_sources=["k"]))


def test_add_placeholder_geolocation_to_dataset(small_detector):
    """use_geo false: every geolocation variable NaN, flags marked not-run, chunked like image_data."""
    ds = _frame_dataset(n_times=6, time_chunk=3)

    result = add_placeholder_geolocation_to_dataset(ds)

    for variable, dtype in _FIELD_VARIABLES.values():
        assert isinstance(result[variable].data, da.Array), variable
        assert result[variable].dtype == dtype, variable
        assert result[variable].shape == (6, *FRAME_SHAPE), variable
        assert result[variable].data.chunks[0] == ds["image_data"].data.chunks[0], variable
    for variable, _ in _PIXEL_VARIABLES.values():
        assert np.isnan(result[variable].values).all(), variable
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


def test_spacecraft_field_mapping_matches_curryer_and_product_definition():
    """The variable maps must name curryer columns and reproduce each product variable's dtype."""
    curryer_columns = {column for field in geolocation._SPACECRAFT_FIELDS for column in field.columns}
    product_variables = yaml.safe_load(product_config_path.read_text())["variables"]

    assert set(geolocation._SCALAR_VARIABLES) <= curryer_columns
    assert set(geolocation._VECTOR_VARIABLES) <= set(geolocation._SPACECRAFT_FIELDS)
    for variable, dtype in [*geolocation._SCALAR_VARIABLES.values(), *geolocation._VECTOR_VARIABLES.values()]:
        assert np.dtype(dtype) == np.dtype(product_variables[variable]["dtype"]), variable

    for variable in [*geolocation._AZIMUTH_CAMERA_TIME_VARIABLES, geolocation._AZIMUTH_INT_TIME_2_VARIABLE]:
        assert np.dtype(product_variables[variable]["dtype"]) == np.float32, variable


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
    # curryer owns the whole-granule coverage check, per SPICE input, so a missing attitude CK
    # raises even while the ephemeris fields are finite.
    assert mock_geometry_data.return_value.get_geometry.call_args.kwargs["require_coverage"] is True
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


@patch("libera_cam.geolocation.spice_error_message", return_value="no coverage for the requested time")
@patch("libera_cam.geolocation.geometry.GeometryData")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1]))
def test_calculate_spacecraft_geometry_raises_friendly_message_on_spice_error(
    mock_adapt, mock_geometry_data, mock_message, mock_kernel_manager, two_timestamps
):
    mock_geometry_data.return_value.get_geometry.side_effect = sp.utils.exceptions.SpiceyError("SPICE(NOFRAMECONNECT)")

    with pytest.raises(RuntimeError, match="no coverage for the requested time"):
        calculate_spacecraft_geometry(mock_kernel_manager, two_timestamps)


@pytest.fixture
def frame_geometry_mocks(spacecraft_geometry_frame):
    """KernelManager, GeometryData, spicetime.adapt and frame_to_frame_euler, patched for two frames."""
    with (
        patch("libera_cam.geolocation.KernelManager") as mock_km_cls,
        patch("libera_cam.geolocation.geometry.GeometryData") as mock_geometry_data,
        patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2])) as mock_adapt,
        patch("libera_cam.geolocation.spatial.frame_to_frame_euler") as mock_euler,
    ):
        mock_km = mock_km_cls.return_value
        mock_km.__enter__.return_value = mock_km
        mock_geometry_data.return_value.get_geometry.return_value = spacecraft_geometry_frame
        mock_euler.side_effect = [_euler_frame([12.5, np.nan]), _euler_frame([13.0, 14.0])]
        yield mock_km_cls, mock_adapt, mock_euler


def test_add_frame_geometry_maps_columns_and_keeps_gaps_nan(frame_geometry_mocks, camera_dataset, caplog):
    """curryer columns map onto product variables; coverage gaps stay NaN and are counted in a warning."""
    mock_km_cls, _, _ = frame_geometry_mocks
    config = GeolocationKernelConfig(dynamic_kernel_sources=["some.bsp"])

    with caplog.at_level(logging.WARNING, logger="libera_cam.geolocation"):
        result = add_frame_geometry_to_dataset(camera_dataset, config)

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert "Subsatellite_Latitude has no SPICE coverage at 1 of 2 camera frames; written as NaN." in warnings
    assert "Satellite_Position has no SPICE coverage at 1 of 2 camera frames; written as NaN." in warnings
    assert "Azimuth_Motor_Position_SPICE_Int_Time_1 has no SPICE coverage at 1 of 2 camera frames; written as NaN." in (
        warnings
    )
    # The second-integration azimuth is covered at both frames.
    assert not any(message.startswith("Azimuth_Motor_Position_SPICE_Int_Time_2") for message in warnings)

    assert result["Subsatellite_Latitude"].dims == ("camera_time",)
    np.testing.assert_allclose(result["Subsatellite_Latitude"].values, [10.0, np.nan])
    np.testing.assert_allclose(result["Subsolar_Longitude"].values, [100.0, np.nan])
    np.testing.assert_allclose(result["Subsatellite_Colatitude"].values, [80.0, np.nan])
    np.testing.assert_allclose(result["Radius_of_Satellite_from_Center_of_Earth"].values, [7000.0, np.nan])
    assert result["Subsatellite_Latitude"].dtype == np.float32
    assert result["Radius_of_Satellite_from_Center_of_Earth"].dtype == np.float64

    assert result["Satellite_Position"].dims == ("camera_time", "EUCLIDEAN_DIM")
    np.testing.assert_allclose(result["Satellite_Position"].values, [[5000.0, 4000.0, 1000.0], [np.nan] * 3])
    np.testing.assert_allclose(result["Satellite_Velocity"].values, [[-2.9, 3.5, 6.0], [np.nan] * 3])
    assert result["Satellite_Position"].dtype == np.float64
    assert result["Satellite_Velocity"].dtype == np.float64

    np.testing.assert_allclose(result["Satellite_Attitude_Q0"].values, [1.0, np.nan])
    np.testing.assert_allclose(result["Satellite_Attitude_Q3"].values, [0.0, np.nan])
    assert result["Satellite_Attitude_Q0"].dtype == np.float32

    # The granule attribute ignores the gap rather than averaging it in.
    assert result.attrs["Earth_Sun_Distance_AU"] == pytest.approx(0.985)

    # One kernel manager furnishes the kernels for every field and releases them within the call.
    mock_km_cls.assert_called_once()
    mock_km = mock_km_cls.return_value
    mock_km.load_libera_dynamic_kernels.assert_called_once_with(
        ["some.bsp"], needs_naif_kernels=True, needs_static_kernels=True
    )
    mock_km.__exit__.assert_called_once()


def test_add_frame_geometry_queries_the_azimuth_ck_at_both_integrations(frame_geometry_mocks, camera_dataset):
    """The CK is read at camera_time (SPICE and Int_Time_1) and 350 ms later (Int_Time_2)."""
    _, mock_adapt, mock_euler = frame_geometry_mocks
    config = GeolocationKernelConfig(dynamic_kernel_sources=["azrot.bc"])

    result = add_frame_geometry_to_dataset(camera_dataset, config)

    assert mock_euler.call_count == 2
    frame_times = pd.DatetimeIndex(camera_dataset.camera_time.values)
    # spicetime.adapt is called for the geometry query, then once per azimuth query.
    pd.testing.assert_index_equal(mock_adapt.call_args_list[1].args[0], frame_times)
    pd.testing.assert_index_equal(mock_adapt.call_args_list[2].args[0], frame_times + pd.Timedelta(350, "ms"))
    for variable in ["Azimuth_Motor_Position_SPICE", "Azimuth_Motor_Position_SPICE_Int_Time_1"]:
        assert result[variable].dims == ("camera_time",)
        assert result[variable].dtype == np.float32
        np.testing.assert_allclose(result[variable].values, [12.5, np.nan])
    np.testing.assert_allclose(result["Azimuth_Motor_Position_SPICE_Int_Time_2"].values, [13.0, 14.0])
    assert result["Azimuth_Motor_Position_SPICE_Int_Time_2"].dtype == np.float32


def test_add_frame_geometry_jpss_only_writes_zero_azimuth(frame_geometry_mocks, camera_dataset):
    """jpss_only has no azimuth CK: the azimuth is 0 degrees and the CK is never read."""
    _, _, mock_euler = frame_geometry_mocks
    config = GeolocationKernelConfig(dynamic_kernel_sources=["jpss.bc"], jpss_only=True)

    result = add_frame_geometry_to_dataset(camera_dataset, config)

    mock_euler.assert_not_called()
    for variable in [*geolocation._AZIMUTH_CAMERA_TIME_VARIABLES, geolocation._AZIMUTH_INT_TIME_2_VARIABLE]:
        assert result[variable].dtype == np.float32
        assert np.all(result[variable].values == 0), variable
    np.testing.assert_allclose(result["Subsatellite_Latitude"].values, [10.0, np.nan])


def test_add_frame_geometry_requires_kernel_sources(camera_dataset):
    with pytest.raises(ValueError, match="SPICE kernel sources are required"):
        add_frame_geometry_to_dataset(camera_dataset, GeolocationKernelConfig())


def test_add_frame_geometry_requires_camera_time():
    with pytest.raises(ValueError, match="camera_time"):
        add_frame_geometry_to_dataset(xr.Dataset(), GeolocationKernelConfig(dynamic_kernel_sources=["k"]))


def test_granule_earth_sun_distance_ignores_gaps():
    assert granule_earth_sun_distance(np.array([np.nan, 0.98, 1.0])) == pytest.approx(0.99)


def test_create_placeholder_spacecraft_geometry():
    result = create_placeholder_spacecraft_geometry(5)

    assert len(result) == 5
    expected_columns = [column for field in geolocation._SPACECRAFT_FIELDS for column in field.columns]
    assert list(result.columns) == expected_columns
    assert np.isnan(result.to_numpy()).all()


def test_add_placeholder_frame_geometry(camera_dataset, caplog):
    with caplog.at_level(logging.WARNING, logger="libera_cam.geolocation"):
        result = add_placeholder_frame_geometry_to_dataset(camera_dataset)

    # Placeholders are NaN by design, not SPICE gaps.
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]

    for variable, dtype in [*geolocation._SCALAR_VARIABLES.values(), *geolocation._VECTOR_VARIABLES.values()]:
        assert result[variable].dtype == dtype, variable
        assert np.isnan(result[variable].values).all(), variable
    assert result["Radius_of_Satellite_from_Center_of_Earth"].dims == ("camera_time",)
    assert result["Satellite_Position"].dims == ("camera_time", "EUCLIDEAN_DIM")
    assert np.isnan(result.attrs["Earth_Sun_Distance_AU"])
    for variable in [*geolocation._AZIMUTH_CAMERA_TIME_VARIABLES, geolocation._AZIMUTH_INT_TIME_2_VARIABLE]:
        assert result[variable].dims == ("camera_time",)
        assert result[variable].dtype == np.float32
        assert np.isnan(result[variable].values).all(), variable


def _euler_frame(euler3: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"euler1": 0.0, "euler2": 0.0, "euler3": euler3}, index=pd.Index(range(len(euler3)), name="ugps")
    )


@patch("libera_cam.geolocation.spatial.frame_to_frame_euler")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2, 3, 4]))
def test_calculate_azimuth_uses_curryer_frame_euler(mock_adapt, mock_euler, mock_kernel_manager):
    """Azimuth is the third 1-2-3 Euler angle of BASE -> AZ, wrapped to [0, 360); CK gaps stay NaN.

    -1e-6 wraps to a float64 just below 360 that rounds to 360.0 in float32, and must be written as 0.
    """
    mock_euler.return_value = _euler_frame([-10.0, 350.0, -1e-6, np.nan])
    timestamps = np.array(
        ["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02", "2025-01-01T00:00:03"],
        dtype="datetime64[ns]",
    )

    azimuth = calculate_azimuth(mock_kernel_manager, timestamps)

    mock_kernel_manager.ensure_known_kernels_are_furnished.assert_called_once()
    call = mock_euler.call_args
    assert call.args[:2] == ("LIBERA_BASE_COORD", "LIBERA_AZ_COORD")
    np.testing.assert_array_equal(call.args[2], [1, 2, 3, 4])
    assert call.kwargs == {"sequence": (1, 2, 3), "allow_nans": True}
    assert azimuth.dtype == np.float32
    np.testing.assert_array_equal(azimuth, np.array([350.0, 350.0, 0.0, np.nan], dtype=np.float32))


@patch("libera_cam.geolocation.spatial.frame_to_frame_euler")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1, 2]))
def test_calculate_azimuth_raises_when_ck_has_no_coverage(mock_adapt, mock_euler, mock_kernel_manager):
    """A granule the CK does not cover at all is a missing input, not a per-frame gap."""
    mock_euler.return_value = _euler_frame([np.nan, np.nan])

    with pytest.raises(ValueError, match="LIBERA_BASE_COORD -> LIBERA_AZ_COORD transform has no coverage"):
        calculate_azimuth(mock_kernel_manager, np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"))
