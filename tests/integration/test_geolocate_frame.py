"""Per-frame geolocation on the DITL_3min data: internal consistency against curryer's boresight fields."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from curryer import spicetime
from curryer.compute import geometry
from curryer.compute.constants import SpatialQualityFlags as SQF
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.constants import GROUND_CAL_PIXEL_MAPPING, PIXEL_COUNT_X, PIXEL_COUNT_Y
from libera_cam.geolocation import geolocate_frame
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data

pytestmark = pytest.mark.integration

_BODY = "LIBERA_WFOV_CAM"


@pytest.fixture
def frame_ugps(test_ditl_l1a_file_path):
    """GPS microseconds of the first DITL frame (kernels loaded separately by each test)."""
    ds = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path, decode_times=True))
    ditl_dir = test_ditl_l1a_file_path.parent
    sources = sorted(p for p in ditl_dir.iterdir() if p.suffix in {".bc", ".bsp"})
    with KernelManager() as km:
        km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        ugps = np.asarray(spicetime.adapt(pd.DatetimeIndex(ds.camera_time.values[:1]), "iso"))
    return ugps, sources


def test_boresight_pixel_matches_curryer_boresight_fields(frame_ugps):
    """A one-pixel frame along the camera boresight reproduces GeometryData's boresight surface angles."""
    ugps, sources = frame_ugps
    fields = [
        geometry.GeometryField.SURFACE_COLATITUDE,
        geometry.GeometryField.VIEWING_ZENITH,
        geometry.GeometryField.SOLAR_ZENITH,
        geometry.GeometryField.VIEWING_AZIMUTH,
        geometry.GeometryField.SOLAR_AZIMUTH,
        geometry.GeometryField.RELATIVE_AZIMUTH,
    ]
    with KernelManager() as km:
        km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        frame = geolocate_frame(ugps, None, _BODY, np.array([[0.0, 0.0, 1.0]]), frame_shape=(1, 1))
        boresight = geometry.GeometryData(_BODY).get_geometry(ugps, fields=fields)

    assert frame.quality_flags[0, 0] == 0
    # Both paths run the same curryer math with no aberration correction; 1e-4 deg is the float32
    # resolution of the frame's values at these magnitudes.
    np.testing.assert_allclose(90.0 - frame.latitude[0, 0], boresight["surface_colatitude"].iloc[0], atol=1e-4)
    for frame_name, column in [
        ("viewing_zenith", "viewing_zenith"),
        ("solar_zenith", "solar_zenith"),
        ("viewing_azimuth", "viewing_azimuth"),
        ("solar_azimuth", "solar_azimuth"),
        ("relative_azimuth", "relative_azimuth"),
    ]:
        np.testing.assert_allclose(
            getattr(frame, frame_name)[0, 0], boresight[column].iloc[0], atol=1e-4, err_msg=column
        )


def test_full_frame_geometry_is_consistent(frame_ugps):
    """Full-frame values: fills off Earth, flags agree with fills, conventions hold, nadir is where it should be."""
    ugps, sources = frame_ugps
    vectors = np.load(GROUND_CAL_PIXEL_MAPPING, mmap_mode="r").reshape(-1, 3)
    with KernelManager() as km:
        km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        frame = geolocate_frame(ugps, None, _BODY, vectors)
        subsatellite = geometry.GeometryData(_BODY).get_geometry(ugps, fields=[geometry.GeometryField.SUBSATELLITE])

    assert frame.latitude.shape == (PIXEL_COUNT_Y, PIXEL_COUNT_X)
    on_earth = frame.latitude != np.float32(-999.0)
    # The square array inscribes the circular field of view; corner pixels point past the limb.
    assert 0.70 < on_earth.mean() < 0.85

    assert np.array_equal(frame.quality_flags == 0, on_earth)
    assert np.all(frame.quality_flags[~on_earth] & SQF.CALC_ELLIPS_NO_INTERSECT)
    for name in ("longitude", "solar_zenith", "solar_azimuth", "viewing_zenith", "viewing_azimuth", "relative_azimuth"):
        assert np.array_equal(getattr(frame, name) != np.float32(-999.0), on_earth), name
    assert np.array_equal(frame.altitude != np.float32(-9999.0), on_earth)
    assert np.all(frame.altitude[on_earth] == 0.0)

    assert np.all((frame.latitude[on_earth] >= -90.0) & (frame.latitude[on_earth] <= 90.0))
    assert np.all((frame.longitude[on_earth] >= -180.0) & (frame.longitude[on_earth] <= 180.0))
    assert np.all((frame.viewing_zenith[on_earth] >= 0.0) & (frame.viewing_zenith[on_earth] <= 90.0))
    assert np.all((frame.solar_zenith[on_earth] >= 0.0) & (frame.solar_zenith[on_earth] <= 180.0))
    for name in ("viewing_azimuth", "solar_azimuth", "relative_azimuth"):
        values = getattr(frame, name)[on_earth]
        assert np.all((values >= 0.0) & (values < 360.0)), name
    # Pins the field mapping and the fill/float32/wrap pipeline, not the convention itself (which
    # the boresight test checks against GeometryData): the relation must survive per pixel.
    # 1e-3 deg covers float32 differencing of two azimuths near 360.
    expected_raa = np.mod(frame.viewing_azimuth[on_earth] - frame.solar_azimuth[on_earth] + 180.0, 360.0)
    shortest_arc = np.mod(frame.relative_azimuth[on_earth] - expected_raa + 180.0, 360.0) - 180.0
    np.testing.assert_allclose(shortest_arc, 0.0, atol=1e-3)

    # Some pixel looks straight down: its viewing zenith is a fraction of a pixel (0.1 deg is
    # under two pixels at 1048 urad) and it sits on the sub-satellite point (0.05 deg is about
    # four times the ground offset a 0.1 deg zenith implies from 824 km).
    nadir = np.unravel_index(np.argmin(np.where(on_earth, frame.viewing_zenith, np.inf)), frame.viewing_zenith.shape)
    assert frame.viewing_zenith[nadir] < 0.1
    np.testing.assert_allclose(frame.latitude[nadir], subsatellite["subsatellite_latitude"].iloc[0], atol=0.05)
    np.testing.assert_allclose(frame.longitude[nadir], subsatellite["subsatellite_longitude"].iloc[0], atol=0.05)
