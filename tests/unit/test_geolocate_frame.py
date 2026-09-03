"""Unit tests for the per-frame geolocation unit, with curryer mocked."""

from unittest.mock import patch

import numpy as np
import pytest
import yaml
from curryer.compute import spatial
from curryer.compute.constants import SpatialQualityFlags as SQF

from libera_cam.config import product_config_path
from libera_cam.geolocation import _AZIMUTH_FIELDS, _PIXEL_VARIABLES, FrameGeometry, geolocate_frame

FRAME_SHAPE = (2, 3)
N_PIXELS = 6
_PIXEL_VARIABLES_CURRYER_NAMES = tuple(curryer_name for curryer_name, *_ in _PIXEL_VARIABLES.values())
EPOCHS = np.array([1_000_000, 2_000_000])
VECTORS = np.tile([0.0, 0.0, 1.0], (N_PIXELS, 1))


def _curryer_result(n_epochs: int) -> spatial.PixelGeometry:
    """Values encode ``100 * epoch + pixel`` so per-pixel epoch selection is visible; pixel 5 misses."""
    base = 100.0 * np.arange(n_epochs)[:, None] + np.arange(N_PIXELS)[None, :]
    base[:, 5] = np.nan
    flags = np.zeros((n_epochs, N_PIXELS), dtype=np.int64)
    flags[:, 5] = int(SQF.CALC_ELLIPS_NO_INTERSECT)
    return spatial.PixelGeometry(
        lon=base + 0.1,
        lat=base + 0.2,
        alt=np.where(np.isnan(base), np.nan, 0.0),
        surface_xyz=np.zeros((n_epochs, N_PIXELS, 3)),
        solar_zenith=base + 0.3,
        solar_azimuth=base + 0.4,
        viewing_zenith=base + 0.5,
        viewing_azimuth=base + 0.6,
        relative_azimuth=base + 0.7,
        quality_flags=flags,
        sc_position=np.zeros((n_epochs, 3)),
        sun_position=np.zeros((n_epochs, 3)),
    )


def test_pixel_variable_mapping_is_bound_to_curryer_and_the_product_definition():
    """Every field maps to a real curryer attribute and to a product variable whose fill and dtype it carries."""
    product_variables = yaml.safe_load(product_config_path.read_text())["variables"]
    assert set(_PIXEL_VARIABLES) == set(FrameGeometry._fields) - {"quality_flags"}
    assert set(_AZIMUTH_FIELDS) <= set(_PIXEL_VARIABLES)
    for curryer_name, product_name, fill, dtype in _PIXEL_VARIABLES.values():
        assert curryer_name in spatial.PixelGeometry._fields
        definition = product_variables[product_name]
        assert definition["dimensions"] == ["CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y"]
        assert np.dtype(definition["dtype"]) == np.dtype(dtype)
        assert definition["attributes"]["_FillValue"] == fill


def test_single_epoch_selects_epoch_zero_and_applies_fills():
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=_curryer_result(1)) as mock_geometry:
        frame = geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=FRAME_SHAPE)

    args = mock_geometry.call_args.args
    np.testing.assert_array_equal(args[0], EPOCHS[:1])
    assert args[1] == "LIBERA_WFOV_CAM"
    np.testing.assert_array_equal(args[2], VECTORS)

    expected = np.arange(6, dtype=np.float64).reshape(FRAME_SHAPE)
    np.testing.assert_allclose(frame.latitude, np.where(expected == 5, -999.0, expected + 0.2), rtol=1e-6)
    np.testing.assert_allclose(frame.longitude, np.where(expected == 5, -999.0, expected + 0.1), rtol=1e-6)
    np.testing.assert_array_equal(frame.altitude, np.where(expected == 5, -9999.0, 0.0))
    np.testing.assert_allclose(frame.relative_azimuth, np.where(expected == 5, -999.0, expected + 0.7), rtol=1e-6)
    for name in FrameGeometry._fields:
        values = getattr(frame, name)
        assert values.shape == FRAME_SHAPE
        assert values.dtype == (np.uint16 if name == "quality_flags" else np.float32), name
    np.testing.assert_array_equal(frame.quality_flags, np.where(expected == 5, SQF.CALC_ELLIPS_NO_INTERSECT, 0))


def test_two_epochs_select_per_pixel_by_exposure_index():
    exposure_index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=_curryer_result(2)):
        frame = geolocate_frame(EPOCHS, exposure_index, "LIBERA_WFOV_CAM", VECTORS, frame_shape=FRAME_SHAPE)

    expected = 100.0 * exposure_index + np.arange(6).reshape(FRAME_SHAPE)
    np.testing.assert_allclose(frame.viewing_zenith, np.where(expected % 100 == 5, -999.0, expected + 0.5), rtol=1e-6)
    np.testing.assert_allclose(frame.solar_azimuth[0], [0.4, 101.4, 2.4], rtol=1e-6)
    np.testing.assert_allclose(frame.solar_azimuth[1], [103.4, 4.4, -999.0], rtol=1e-6)


def test_azimuths_just_under_360_wrap_to_zero_after_float32_cast():
    """359.99999 in float64 is 360.0 in float32; curryer's azimuth convention is the half-open [0, 360)."""
    result = _curryer_result(1)
    nearly_full_turn = np.where(np.isnan(result.lat), np.nan, 359.9999999)
    result = result._replace(
        solar_azimuth=nearly_full_turn, viewing_azimuth=nearly_full_turn, relative_azimuth=nearly_full_turn
    )
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result):
        frame = geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=FRAME_SHAPE)

    assert np.float32(359.9999999) == np.float32(360.0)
    for name in ("solar_azimuth", "viewing_azimuth", "relative_azimuth"):
        values = getattr(frame, name)
        np.testing.assert_array_equal(values[0], [0.0, 0.0, 0.0], err_msg=name)
        np.testing.assert_array_equal(values[1], [0.0, 0.0, -999.0], err_msg=name)
    # Zeniths are not wrapped.
    assert frame.solar_zenith[0, 0] == np.float32(0.3)


def test_altitude_is_converted_from_kilometers_to_meters():
    result = _curryer_result(1)
    result = result._replace(alt=np.where(np.isnan(result.alt), np.nan, 1.5))
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result):
        frame = geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=FRAME_SHAPE)
    assert frame.altitude[0, 0] == np.float32(1500.0)


@pytest.mark.parametrize(
    ("epochs", "index", "vectors", "match"),
    [
        (np.array([[1, 2]]), None, VECTORS, "non-empty 1-D"),
        (np.array([]), None, VECTORS, "non-empty 1-D"),
        (EPOCHS[:1], None, VECTORS[:4], "pointing_vectors must be shape"),
        (EPOCHS[:1], np.zeros(FRAME_SHAPE, dtype=int), VECTORS, "must be None"),
        (EPOCHS, None, VECTORS, "exposure_index is required"),
        (EPOCHS, np.zeros((3, 2), dtype=int), VECTORS, "shaped"),
        (EPOCHS, np.zeros(FRAME_SHAPE, dtype=float), VECTORS, "integer array"),
        (EPOCHS, np.full(FRAME_SHAPE, 2, dtype=int), VECTORS, "must lie in"),
        (EPOCHS, np.full(FRAME_SHAPE, -1, dtype=int), VECTORS, "must lie in"),
    ],
)
def test_geolocate_frame_rejects_bad_inputs(epochs, index, vectors, match):
    with patch("libera_cam.geolocation.spatial.pixel_geometry") as mock_geometry:
        with pytest.raises(ValueError, match=match):
            geolocate_frame(epochs, index, "LIBERA_WFOV_CAM", vectors, frame_shape=FRAME_SHAPE)
    mock_geometry.assert_not_called()


@pytest.mark.parametrize("body", ["LIBERA_RAD", "JPSS4_SC", "LIBERA_AZ", ""])
def test_geolocate_frame_rejects_non_camera_bodies(body):
    """A valid SPICE body in the wrong role would geolocate plausibly; only the two camera bodies pass."""
    with patch("libera_cam.geolocation.spatial.pixel_geometry") as mock_geometry:
        with pytest.raises(ValueError, match="camera observer"):
            geolocate_frame(EPOCHS[:1], None, body, VECTORS, frame_shape=FRAME_SHAPE)
    mock_geometry.assert_not_called()


def test_jpss_only_body_is_accepted():
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=_curryer_result(1)) as mock_geometry:
        geolocate_frame(EPOCHS[:1], None, "LIBERA_BASE", VECTORS, frame_shape=FRAME_SHAPE)
    assert mock_geometry.call_args.args[1] == "LIBERA_BASE"


def test_uncovered_epoch_keeps_its_compound_flag_word():
    """A SPICE gap fills the whole epoch and the compound curryer flag survives the uint16 narrowing."""
    result = _curryer_result(2)
    gap = int(SQF.SPICE_ERR_MISSING_ATTITUDE | SQF.CALC_ELLIPS_INSUFF_DATA)
    result.quality_flags[1, :] = gap
    # curryer leaves every per-pixel value of an uncovered epoch NaN.
    gap_row = np.array([[False] * 6, [True] * 6])
    result = result._replace(
        **{name: np.where(gap_row, np.nan, getattr(result, name)) for name in _PIXEL_VARIABLES_CURRYER_NAMES}
    )
    exposure_index = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    with patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result):
        frame = geolocate_frame(EPOCHS, exposure_index, "LIBERA_WFOV_CAM", VECTORS, frame_shape=FRAME_SHAPE)

    assert gap == 0x104
    np.testing.assert_array_equal(frame.quality_flags[0], [0, gap, 0])
    # An uncovered epoch carries the gap word alone: no intersection was attempted for it.
    np.testing.assert_array_equal(frame.quality_flags[1], [gap, 0, gap])
    np.testing.assert_allclose(frame.latitude[0], [0.2, -999.0, 2.2], rtol=1e-6)
    np.testing.assert_allclose(frame.latitude[1], [-999.0, 4.2, -999.0], rtol=1e-6)
    for name in FrameGeometry._fields:
        if name != "quality_flags":
            fill = -9999.0 if name == "altitude" else -999.0
            np.testing.assert_array_equal(getattr(frame, name)[exposure_index == 1], fill, err_msg=name)


def test_geolocate_frame_rejects_flags_beyond_uint16():
    result = _curryer_result(1)
    result.quality_flags[0, 0] = 1 << 16
    with (
        patch("libera_cam.geolocation.spatial.pixel_geometry", return_value=result),
        pytest.raises(RuntimeError, match="uint16"),
    ):
        geolocate_frame(EPOCHS[:1], None, "LIBERA_WFOV_CAM", VECTORS, frame_shape=FRAME_SHAPE)
