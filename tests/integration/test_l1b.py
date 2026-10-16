"""Tests for the l1b algorithm"""

import logging
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from libera_utils.io.manifest import Manifest, ManifestType

from libera_cam.l1b import algorithm
from libera_cam.version import version as libera_cam_version

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def input_manifest_path(tmp_path_factory, test_data_path):
    """Generate a test input manifest from the DITL_3min integration data."""
    ditl_data_path = test_data_path / "DITL_3min"

    filenames = (
        ditl_data_path / "LIBERA_L1A_WFOV-SCI-DECODED_V5-10-6_20280212T034136_20280212T034141_R26247171104.nc",
        ditl_data_path / "LIBERA_SPICE_AZROT-CK_V5-8-4_20280212T033106_20280212T040027_R26157131711.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20280212T000000_20280212T235959_R26006200637.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280212T000000_20280212T235959_R26006200633.bsp",
    )

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=filenames)
    return input_manifest.write(tmp_path_factory.mktemp("input_manifest"))


@pytest.fixture(scope="module")
def l1b_product_file_path(input_manifest_path, tmp_path_factory):
    """Run the L1B algorithm once per test module and return the output data file path."""
    tmp_path = tmp_path_factory.mktemp("l1b_product")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PROCESSING_PATH", str(tmp_path))
        output_manifest_path = algorithm(Namespace(manifest=str(input_manifest_path)))

    output_manifest = Manifest.from_file(output_manifest_path)
    nc_files = [file for file in output_manifest.files if Path(file.filename).suffix == ".nc"]
    assert len(nc_files) == 1, "Expected exactly one L1B output file in manifest"
    return nc_files[0].filename


@pytest.fixture(scope="module")
def l1b_product_dataset(l1b_product_file_path):
    """Open the L1B output for science and invariant checks."""
    with xr.open_dataset(l1b_product_file_path) as ds:
        return ds.load()


def test_algorithm(l1b_product_dataset):
    """Testing the algorithm to generate output manifests"""
    print(l1b_product_dataset)


def test_algorithm_version_matches_package(l1b_product_dataset):
    """The algorithm_version global attribute must match the installed package version."""
    assert l1b_product_dataset.attrs["algorithm_version"] == libera_cam_version()


def test_earth_sun_distance_plausible(l1b_product_dataset):
    """The granule Earth-Sun distance must land within Earth's orbital range."""
    distance_au = float(l1b_product_dataset.attrs["Earth_Sun_Distance_AU"])
    assert 0.95 <= distance_au <= 1.05, f"Earth_Sun_Distance_AU={distance_au:.4f} is outside [0.95, 1.05]"


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
def test_spacecraft_geometry_is_computed_per_frame(l1b_product_dataset, variable, low, high):
    """Each spacecraft-level field is one covered value per camera frame, inside its valid range.

    xarray decodes the written ``_FillValue`` to NaN on read, so fill shows up as NaN here.
    """
    data_array = l1b_product_dataset[variable]
    assert data_array.dims == ("CAMERA_TIME",)

    values = data_array.to_numpy()
    assert np.isfinite(values).all(), f"{variable} contains _FillValue; SPICE coverage was expected"
    assert values.min() >= low
    assert values.max() <= high


def test_satellite_state_vectors_are_per_frame_inertial(l1b_product_dataset):
    """Position and velocity are covered (N, 3) J2000 vectors consistent with the geocentric radius."""
    position = l1b_product_dataset["Satellite_Position"]
    velocity = l1b_product_dataset["Satellite_Velocity"]
    assert position.dims == ("CAMERA_TIME", "EUCLIDEAN_DIM")
    assert velocity.dims == ("CAMERA_TIME", "EUCLIDEAN_DIM")
    assert np.isfinite(position.to_numpy()).all()
    assert np.isfinite(velocity.to_numpy()).all()

    # The norm is frame-invariant, so the J2000 position must reproduce the ECEF-derived radius.
    radius = l1b_product_dataset["Radius_of_Satellite_from_Center_of_Earth"].to_numpy()
    np.testing.assert_allclose(np.linalg.norm(position.to_numpy(), axis=1), radius, atol=1e-3)

    # Low-Earth-orbit speed.
    speed = np.linalg.norm(velocity.to_numpy(), axis=1)
    assert speed.min() >= 7.0
    assert speed.max() <= 8.0


def test_satellite_attitude_is_unit_quaternion(l1b_product_dataset):
    components = np.stack([l1b_product_dataset[f"Satellite_Attitude_Q{i}"].to_numpy() for i in range(4)], axis=1)
    np.testing.assert_allclose(np.linalg.norm(components, axis=1), 1.0, atol=1e-5)


def test_subsatellite_colatitude_complements_latitude(l1b_product_dataset):
    """Colatitude is the geodetic complement of latitude, so the two must stay consistent."""
    latitude = l1b_product_dataset["Subsatellite_Latitude"].to_numpy()
    colatitude = l1b_product_dataset["Subsatellite_Colatitude"].to_numpy()
    np.testing.assert_allclose(colatitude, 90.0 - latitude, atol=1e-4)


def test_three_dimensional_variables_carry_the_pinned_encoding_on_disk(l1b_product_file_path):
    """The shuffle filter and chunk shape declared in the product definition must reach the written file."""
    with h5py.File(l1b_product_file_path, "r") as product:
        for name in ("Latitude", "Longitude", "Viewing_Zenith_Surface", "Geolocation_Quality_Flag"):
            assert product[name].shuffle is True, name
        # Pinned off, against the netcdf4 engine's default of on whenever zlib is on.
        assert product["Radiance"].shuffle is False
        assert product["Pixel_Counts"].shuffle is True
        for name in ("Latitude", "Longitude", "Viewing_Zenith_Surface", "Geolocation_Quality_Flag", "Radiance"):
            assert product[name].chunks == (1, 512, 512), name


def test_per_pixel_geometry_is_computed_for_every_kept_pixel(l1b_product_dataset):
    """Kept on-Earth pixels have finite geometry flagged 0; pixels the camera removed are fill with bit 15.

    xarray decodes the written ``_FillValue`` to NaN on read; the flag variable declares no fill and
    keeps its uint16 dtype. DITL_3min mixes masked frames (about 8% kept) with an unmasked one.
    """
    flags = l1b_product_dataset["Geolocation_Quality_Flag"]
    assert flags.dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    assert flags.dtype == np.uint16
    kept = l1b_product_dataset["Camera_Mask"].to_numpy() == 1
    assert "valid_pixel_mask" not in l1b_product_dataset

    flag_values = flags.to_numpy()
    on_earth = flag_values == 0
    assert on_earth.any(axis=(1, 2)).all(), "every frame has geolocated pixels"
    # On an unmasked frame the square detector inscribes the circular field of view; corner pixels
    # point past the limb.
    unmasked = kept.all(axis=(1, 2))
    assert unmasked.any(), "DITL_3min has an unmasked frame"
    on_earth_fraction = on_earth[unmasked].mean(axis=(1, 2))
    assert np.all((on_earth_fraction > 0.70) & (on_earth_fraction < 0.85)), on_earth_fraction
    assert not on_earth[~kept].any()
    np.testing.assert_array_equal(flag_values[~kept], 0x8000)
    assert np.isnan(l1b_product_dataset["Radiance"].to_numpy()[~kept]).all()
    for name in (
        "Latitude",
        "Longitude",
        "Altitude",
        "Solar_Zenith_Surface",
        "Viewing_Zenith_Surface",
        "Relative_Azimuth_Surface",
        "Viewing_Azimuth_Surface_WRT_North",
        "Solar_Azimuth_Surface_WRT_North",
    ):
        assert np.array_equal(np.isfinite(l1b_product_dataset[name].to_numpy()), on_earth), name


def test_azimuth_is_covered_motor_encoder_angle(l1b_product_dataset):
    """Azimuth is one covered float32 per frame from the AZROT-CK, in [0, 360), tracking the FSW header.

    The FSW header ``Azimuth_FSW_Metadata`` records the same encoder angle, signed in degrees: on the
    DITL_3min frames it reads 356.05 and 353.54 degrees (wrapped) against the CK's 356.15 and 353.63.
    Agreement within 0.5 degrees pins the CK sign, frame order and Euler component; reversing any of
    them moves the difference to several degrees.
    """
    azimuth = l1b_product_dataset["Azimuth"]
    assert azimuth.dims == ("CAMERA_TIME",)
    assert azimuth.dtype == np.float32

    values = azimuth.to_numpy()
    assert np.isfinite(values).all(), "Azimuth contains _FillValue; AZROT-CK coverage was expected"
    assert np.all((values >= 0.0) & (values < 360.0))

    header = l1b_product_dataset["Azimuth_FSW_Metadata"].to_numpy().astype(np.float64)
    wrapped_difference = (values.astype(np.float64) - header + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(wrapped_difference, 0.0, atol=0.5)


def _write_product(manifest_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the L1B algorithm into ``output_dir`` and return the written NetCDF path."""
    output_dir.mkdir()
    monkeypatch.setenv("PROCESSING_PATH", str(output_dir))
    output_manifest = Manifest.from_file(algorithm(Namespace(manifest=str(manifest_path))))
    (nc_file,) = [file.filename for file in output_manifest.files if Path(file.filename).suffix == ".nc"]
    return nc_file


def _dataset_names(h5file: h5py.File) -> list[str]:
    names = []
    h5file.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    return sorted(names)


@pytest.mark.parametrize("configuration", [{}, {"use_geo": False}], ids=["use_geo", "no_geo"])
def test_distributed_product_matches_synchronous(generate_input_manifest, monkeypatch, tmp_path, caplog, configuration):
    """The worker-compressed write under distributed stores the same bits, chunks and filters as synchronous.

    Under distributed the workers shuffle and deflate every Dask-backed 3-D variable's chunks and the
    client stores them with ``write_direct_chunk``; under synchronous HDF5 compresses as it writes.
    """
    manifest_path = generate_input_manifest(configuration)
    synchronous_path = _write_product(manifest_path, tmp_path / "synchronous", monkeypatch)

    monkeypatch.setenv("DASK_SCHEDULER", "distributed")
    monkeypatch.setenv("DASK_NUM_WORKERS", "1")
    monkeypatch.setenv("DASK_MEMORY_LIMIT", "6GB")
    with caplog.at_level(logging.INFO, logger="libera_utils.io.netcdf"):
        distributed_path = _write_product(manifest_path, tmp_path / "distributed", monkeypatch)
    assert "worker-compressed chunks" in caplog.text

    with h5py.File(synchronous_path, "r") as expected, h5py.File(distributed_path, "r") as actual:
        names = _dataset_names(expected)
        assert _dataset_names(actual) == names
        for name in names:
            want, got = expected[name], actual[name]
            layout = ("dtype", "shape", "chunks", "compression", "compression_opts", "shuffle")
            assert [getattr(got, key) for key in layout] == [getattr(want, key) for key in layout], name
            assert np.asarray(got.fillvalue).tobytes() == np.asarray(want.fillvalue).tobytes(), name
            want_values, got_values = np.asarray(want[()]), np.asarray(got[()])
            bits = f"u{want_values.dtype.itemsize}"
            np.testing.assert_array_equal(got_values.view(bits), want_values.view(bits), err_msg=name)
