"""Integration tests for the lazy per-pixel geolocation on the DITL_3min data."""

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from curryer import spicetime
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.constants import GROUND_CAL_PIXEL_MAPPING
from libera_cam.geolocation import (
    _FIELD_VARIABLES,
    GeolocationKernelConfig,
    add_frame_geometry_to_dataset,
    add_geolocation_to_dataset,
    geolocate_frame,
)
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data

pytestmark = pytest.mark.integration


def _ditl_dynamic_kernel_sources(test_data_path: Path) -> list[Path]:
    d = test_data_path / "DITL_3min"
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"})


def test_add_geolocation_to_dataset_reproduces_geolocate_frame(test_data_path, test_ditl_l1a_file_path, monkeypatch):
    """The lazy path splits frames into geometry tasks and writes exactly what geolocate_frame returns."""
    monkeypatch.setenv("LIBERA_CAM_GEO_CHUNK_SIZE", "2")
    ds = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path)).isel(camera_time=slice(0, 3))
    sources = _ditl_dynamic_kernel_sources(test_data_path)
    config = GeolocationKernelConfig(dynamic_kernel_sources=sources)

    ds_geo = add_geolocation_to_dataset(ds, config)

    variables = [variable for variable, _ in _FIELD_VARIABLES.values()]
    for variable, dtype in _FIELD_VARIABLES.values():
        assert isinstance(ds_geo[variable].data, da.Array), variable
        assert ds_geo[variable].dims == ("camera_time", "y", "x")
        assert ds_geo[variable].dtype == dtype, variable
        assert ds_geo[variable].data.chunks[0] == (2, 1), variable

    # The last frame sits alone in the second task.
    computed = ds_geo[variables].isel(camera_time=2).compute(scheduler="synchronous")
    with KernelManager() as km:
        km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        ugps = np.asarray(spicetime.adapt(pd.DatetimeIndex(ds.camera_time.values[2:3]), "iso"))
        vectors = np.load(GROUND_CAL_PIXEL_MAPPING, mmap_mode="r").reshape(-1, 3)
        expected = geolocate_frame(ugps, None, "LIBERA_WFOV_CAM", vectors)

    for name, (variable, _) in _FIELD_VARIABLES.items():
        np.testing.assert_array_equal(computed[variable].values, getattr(expected, name), err_msg=variable)
    on_earth = computed["Geolocation_Quality_Flag"].values == 0
    assert 0.70 < on_earth.mean() < 0.85


def test_add_geolocation_rejects_a_granule_the_kernels_do_not_cover(test_data_path, test_ditl_l1a_file_path):
    """Frames a day past the kernels: no frame is covered, so the granule is rejected before any task runs."""
    ds = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path)).isel(camera_time=slice(0, 2))
    ds = ds.assign_coords(camera_time=ds.camera_time + np.timedelta64(1, "D"))
    config = GeolocationKernelConfig(dynamic_kernel_sources=_ditl_dynamic_kernel_sources(test_data_path))

    with pytest.raises(RuntimeError, match="cover none of the 2 camera frame"):
        add_geolocation_to_dataset(ds, config)


def test_dynamic_kernel_sequence_materializes_into_cache(monkeypatch, tmp_path, test_data_path):
    """Explicit sequence mode should materialize kernel basenames into the user cache via KernelFileCache."""
    test_kernel_dir = test_data_path / "DITL_3min"
    kernel_files = sorted(
        [p for p in test_kernel_dir.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"}],
        key=lambda p: p.name,
    )
    assert kernel_files, f"No dynamic kernels found under {test_kernel_dir}"
    sources = kernel_files[:2]

    monkeypatch.setattr(spice_utils.caching, "get_local_cache_dir", lambda: tmp_path)

    km = KernelManager(cache_timeout_days=7)
    km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)

    for src in sources:
        assert (tmp_path / src.name).is_file(), f"Expected cached kernel missing: {src.name}"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("shift", "match"),
    [
        # Outside every DITL kernel: the spacecraft geometry query raises first.
        (np.timedelta64(30, "D"), "returned all-NaN"),
        # Inside the day-long JPSS SPK and CK, outside the AZROT-CK (03:31 to 04:00).
        (np.timedelta64(2, "h"), "has no coverage over any of the"),
    ],
)
def test_granule_outside_kernel_coverage_raises(test_data_path, test_ditl_l1a_file_path, shift, match):
    """A granule the kernels cover at no frame stops the run rather than writing an all-NaN product."""
    ds = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path))
    ds = ds.assign_coords(camera_time=ds.camera_time + shift)
    config = GeolocationKernelConfig(dynamic_kernel_sources=_ditl_dynamic_kernel_sources(test_data_path))

    with pytest.raises(ValueError, match=match):
        add_frame_geometry_to_dataset(ds, config)


@pytest.fixture(scope="module")
def ditl_frame_geometry(test_data_path):
    """DITL_3min camera frames with the per-frame spacecraft geometry and motor azimuth from the DITL kernels."""
    l1a_path = (
        test_data_path
        / "DITL_3min"
        / "LIBERA_L1A_WFOV-SCI-DECODED_V5-10-6_20280212T034136_20280212T034141_R20260904T171104000000.nc"
    )
    with xr.open_dataset(l1a_path) as l1a_ds:
        ds = read_l1a_cam_data(l1a_ds)
        config = GeolocationKernelConfig(dynamic_kernel_sources=_ditl_dynamic_kernel_sources(test_data_path))
        return add_frame_geometry_to_dataset(ds, config).compute()


@pytest.mark.integration
def test_earth_sun_distance_plausible(ditl_frame_geometry):
    """The granule Earth-Sun distance must land within Earth's orbital range."""
    distance_au = float(ditl_frame_geometry.attrs["Earth_Sun_Distance_AU"])
    assert 0.95 <= distance_au <= 1.05, f"Earth_Sun_Distance_AU={distance_au:.4f} is outside [0.95, 1.05]"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("variable", "low", "high"),
    [
        ("Subsatellite_Latitude", -90.0, 90.0),
        ("Subsatellite_Longitude", -180.0, 180.0),
        ("Subsatellite_Colatitude", 0.0, 180.0),
        ("Subsolar_Latitude", -90.0, 90.0),
        ("Subsolar_Longitude", -180.0, 180.0),
        ("Subsolar_Colatitude", 0.0, 180.0),
        ("Radius_of_Satellite_from_Center_of_Earth", 6000.0, 8000.0),
        ("Satellite_Attitude_Q0", -1.0, 1.0),
        ("Satellite_Attitude_Q1", -1.0, 1.0),
        ("Satellite_Attitude_Q2", -1.0, 1.0),
        ("Satellite_Attitude_Q3", -1.0, 1.0),
    ],
)
def test_spacecraft_geometry_within_valid_range(ditl_frame_geometry, variable, low, high):
    """Each spacecraft-level field is covered at every DITL frame and inside its valid range."""
    values = ditl_frame_geometry[variable].to_numpy()
    assert np.isfinite(values).all(), f"{variable} contains NaN; SPICE coverage was expected"
    assert values.min() >= low
    assert values.max() <= high


@pytest.mark.integration
def test_satellite_state_vectors_match_radius_and_leo_speed(ditl_frame_geometry):
    """The J2000 position reproduces the geocentric radius, and the speed is a low-Earth-orbit speed."""
    position = ditl_frame_geometry["Satellite_Position"].to_numpy()
    velocity = ditl_frame_geometry["Satellite_Velocity"].to_numpy()

    # The norm is frame-invariant, so the J2000 position must reproduce the ECEF-derived radius.
    radius = ditl_frame_geometry["Radius_of_Satellite_from_Center_of_Earth"].to_numpy()
    np.testing.assert_allclose(np.linalg.norm(position, axis=1), radius, atol=1e-3)

    speed = np.linalg.norm(velocity, axis=1)
    assert speed.min() >= 7.0
    assert speed.max() <= 8.0


@pytest.mark.integration
def test_satellite_attitude_is_unit_quaternion(ditl_frame_geometry):
    components = np.stack([ditl_frame_geometry[f"Satellite_Attitude_Q{i}"].to_numpy() for i in range(4)], axis=1)
    np.testing.assert_allclose(np.linalg.norm(components, axis=1), 1.0, atol=1e-5)


@pytest.mark.integration
def test_subsatellite_colatitude_complements_latitude(ditl_frame_geometry):
    """Colatitude is the geodetic complement of latitude, so the two must stay consistent."""
    latitude = ditl_frame_geometry["Subsatellite_Latitude"].to_numpy()
    colatitude = ditl_frame_geometry["Subsatellite_Colatitude"].to_numpy()
    np.testing.assert_allclose(colatitude, 90.0 - latitude, atol=1e-4)


@pytest.mark.integration
def test_azimuth_at_first_integration_tracks_fsw_header(ditl_frame_geometry):
    """The CK azimuth at camera_time tracks the FSW header azimuth.

    The FSW header records the same encoder angle, signed in degrees: on the DITL_3min frames it
    reads 356.05 and 353.54 degrees (wrapped) against the CK's 356.15 and 353.63. Agreement within
    0.5 degrees pins the CK sign, frame order and Euler component; reversing any of them moves the
    difference to several degrees.
    """
    azimuth = ditl_frame_geometry["Azimuth_Motor_Position_SPICE_Int_Time_1"].to_numpy()
    assert np.isfinite(azimuth).all(), "AZROT-CK coverage was expected at every frame"
    assert np.all((azimuth >= 0.0) & (azimuth < 360.0))
    np.testing.assert_array_equal(ditl_frame_geometry["Azimuth_Motor_Position_SPICE"].to_numpy(), azimuth)

    header = ditl_frame_geometry["azimuth_angle"].to_numpy().astype(np.float64)
    wrapped_difference = (azimuth.astype(np.float64) - header + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(wrapped_difference, 0.0, atol=0.5)


@pytest.mark.integration
def test_azimuth_at_second_integration_is_covered(ditl_frame_geometry):
    """The CK covers camera_time + 350 ms at every DITL frame, and the angle stays in [0, 360)."""
    azimuth = ditl_frame_geometry["Azimuth_Motor_Position_SPICE_Int_Time_2"].to_numpy()
    assert azimuth.dtype == np.float32
    assert np.isfinite(azimuth).all()
    assert np.all((azimuth >= 0.0) & (azimuth < 360.0))
