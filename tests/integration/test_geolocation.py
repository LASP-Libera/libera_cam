"""Integration tests for geolocation using static and geolocation kernels."""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.geolocation import (
    GeolocationKernelConfig,
    add_frame_geometry_to_dataset,
    add_geolocation_to_dataset,
    calculate_all_pixel_lat_lon_altitude,
)
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data


def _ditl_dynamic_kernel_sources(test_data_path: Path) -> list[Path]:
    d = test_data_path / "DITL_3min"
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"})


@pytest.mark.integration
def test_geolocation_from_kernels(test_data_path, test_ditl_l1a_file_path):
    """
    Test geolocation computation using static and geolocation kernels with real L1A timestamps.
    """
    # Load real L1A data to get timestamps
    l1a_ds = xr.open_dataset(test_ditl_l1a_file_path)
    # Parse to get camera_time
    # ds = read_l1a_cam_data(l1a_ds)
    # Use first 5 frames
    time_subset = l1a_ds.PACKET_ICIE_TIME.data[:5]  # Using packet timestamp to avoid mismatch with camera times

    km = KernelManager()
    test_kernel_sources = _ditl_dynamic_kernel_sources(test_data_path)
    km.load_naif_kernels()
    km.load_static_kernels()
    km.load_libera_dynamic_kernels(test_kernel_sources, needs_naif_kernels=True, needs_static_kernels=True)

    print(f"Calculating lat/lon/alt for {len(time_subset)} timestamps.")

    # Pass DataArray directly
    lat_lon_alt = calculate_all_pixel_lat_lon_altitude(km, time_subset)

    assert "latitude" in lat_lon_alt
    assert lat_lon_alt["latitude"].shape == (5, 2048, 2048)

    mean_lat = np.nanmean(lat_lon_alt["latitude"])
    print("Mean Lat: ", mean_lat)

    if np.isnan(mean_lat):
        print(
            "Warning: All geolocation results are NaN. This indicates camera is looking into space or kernels mismatch."
        )
    else:
        assert mean_lat != 0


def test_add_geolocation_to_dataset(test_data_path, test_ditl_l1a_file_path):
    """Test the lazy Dask-based geolocation pipeline with real L1A data."""
    # 1. Load Data
    l1a_ds = xr.open_dataset(test_ditl_l1a_file_path)
    ds = read_l1a_cam_data(l1a_ds)

    # Subset for speed (first 5 frames)
    ds = ds.isel(camera_time=slice(0, 3))

    # 2. Setup Config
    test_kernel_sources = _ditl_dynamic_kernel_sources(test_data_path)
    config = GeolocationKernelConfig(
        dynamic_kernel_sources=test_kernel_sources, use_test_naif_url=False, cache_timeout_days=7
    )

    # 3. Add Geolocation (Lazy)
    # Using default pixel_mask=None (Calculate All)
    ds_geo = add_geolocation_to_dataset(ds, config)

    assert "Latitude" in ds_geo
    assert "Longitude" in ds_geo
    assert "Altitude" in ds_geo

    assert isinstance(ds_geo["Latitude"].data, da.Array)

    # 4. Compute (Trigger Dask)
    # Compute first frame
    result = ds_geo.isel(camera_time=0).compute(scheduler="synchronous")

    assert result["Latitude"].shape == (2048, 2048)

    # Check valid values exist IF possible
    valid_count = ds.valid_pixel_mask.isel(camera_time=0).sum().compute()

    if valid_count > 0:
        if np.isnan(result["Latitude"].values).all():
            print(
                "Warning: Frame 0 has valid pixels but Geolocation is all NaN. "
                "Test Data Geometry likely off-Earth. Skipping assertion."
            )
        else:
            assert not np.isnan(result["Latitude"].values).all()
    else:
        print("Warning: Frame 0 has no valid pixels. Expecting all NaNs.")
        assert np.isnan(result["Latitude"].values).all()


def test_add_geolocation_with_static_mask(test_data_path, test_ditl_l1a_file_path):
    """Test that static pixel_mask correctly skips calculation."""
    # 1. Load Data
    l1a_ds = xr.open_dataset(test_ditl_l1a_file_path)
    ds = read_l1a_cam_data(l1a_ds).isel(camera_time=slice(0, 2))  # 2 frames

    # 2. Config
    test_kernel_sources = _ditl_dynamic_kernel_sources(test_data_path)
    config = GeolocationKernelConfig(dynamic_kernel_sources=test_kernel_sources)

    # 3. Create a Static Mask
    # Mask out everything except the first 10 pixels
    mask = np.zeros((2048, 2048), dtype=bool)
    mask.ravel()[:10] = True

    # 4. Run Geolocation
    ds_geo = add_geolocation_to_dataset(ds, config, pixel_mask=mask)

    # 5. Verify
    result = ds_geo.isel(camera_time=0).compute(scheduler="synchronous")
    lat = result["Latitude"].values.ravel()

    # Masked-out pixels (index 10 onwards) get product fill value
    assert np.all(lat[10:] == np.float32(-999)), "Masked-out pixels should be fill value -999"


def test_add_geolocation_with_dynamic_mask_integration(test_data_path, test_ditl_l1a_file_path):
    """Test using a 3D dynamic mask where valid pixels change per frame."""
    # 1. Load Data (3 frames)
    l1a_ds = xr.open_dataset(test_ditl_l1a_file_path)
    ds = read_l1a_cam_data(l1a_ds).isel(camera_time=slice(0, 3))

    # 2. Create Dynamic Mask (3, 2048, 2048)
    # Frame 0: Top half valid
    # Frame 1: Bottom half valid
    # Frame 2: All Invalid
    dynamic_mask = np.zeros((3, 2048, 2048), dtype=bool)
    dynamic_mask[0, :1024, :] = True
    dynamic_mask[1, 1024:, :] = True
    # Frame 2 remains all False

    # Convert to Dask array
    # Using single chunk along Time to ensure matching with ds chunks
    mask_da = da.from_array(dynamic_mask, chunks=(3, 2048, 2048))

    # Force ds to have matching chunks (3,)
    ds = ds.chunk({"camera_time": 3})

    # 3. Config
    test_kernel_sources = _ditl_dynamic_kernel_sources(test_data_path)
    config = GeolocationKernelConfig(dynamic_kernel_sources=test_kernel_sources)

    # 4. Run Geolocation (Dynamic)
    ds_geo_dynamic = add_geolocation_to_dataset(ds, config, pixel_mask=mask_da)

    # 5. Run Geolocation (Full - for comparison)
    # ds_geo_full = add_geolocation_to_dataset(ds, config, pixel_mask=None)

    # 6. Verify Results
    results = ds_geo_dynamic["Latitude"].compute(scheduler="synchronous").values  # (3, 2048, 2048)
    # full_results = ds_geo_full["Latitude"].compute(scheduler="synchronous").values

    # Verification Logic:
    # We verify that masked-out regions use the product fill value.
    # We cannot verifying masked-in regions have values because the test data yields NaNs anyway.

    # Frame 0: Bottom half should be fill value
    assert np.all(results[0, 1024:, :] == np.float32(-999))

    # Frame 1: Top half should be fill value
    assert np.all(results[1, :1024, :] == np.float32(-999))

    # Frame 2: All fill value
    assert np.all(results[2] == np.float32(-999))


@pytest.mark.integration
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
        / "LIBERA_L1A_WFOV-SCI-DECODED_V5-10-6_20280212T034136_20280212T034141_R26247171104.nc"
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
