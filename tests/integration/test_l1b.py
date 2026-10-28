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
        ditl_data_path
        / "LIBERA_L1A_WFOV-SCI-DECODED_V5-10-6_20280212T034136_20280212T034141_R20260904T171104000000.nc",
        ditl_data_path / "LIBERA_SPICE_AZROT-CK_V5-8-4_20280212T033106_20280212T040027_R20260606T131711000000.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20280212T000000_20280212T235959_R20260106T200637000000.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280212T000000_20280212T235959_R20260106T200633000000.bsp",
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
    nc_files = [file for file in output_manifest.files if Path(file).suffix == ".nc"]
    assert len(nc_files) == 1, "Expected exactly one L1B output file in manifest"
    return nc_files[0]


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


@pytest.mark.parametrize(
    ("variable", "dims"),
    [
        ("Subsatellite_Latitude", ("CAMERA_TIME",)),
        ("Subsatellite_Longitude", ("CAMERA_TIME",)),
        ("Subsatellite_Colatitude", ("CAMERA_TIME",)),
        ("Subsolar_Latitude", ("CAMERA_TIME",)),
        ("Subsolar_Longitude", ("CAMERA_TIME",)),
        ("Subsolar_Colatitude", ("CAMERA_TIME",)),
        ("Radius_of_Satellite_from_Center_of_Earth", ("CAMERA_TIME",)),
        ("Satellite_Attitude_Q0", ("CAMERA_TIME",)),
        ("Satellite_Attitude_Q1", ("CAMERA_TIME",)),
        ("Satellite_Attitude_Q2", ("CAMERA_TIME",)),
        ("Satellite_Attitude_Q3", ("CAMERA_TIME",)),
        ("Satellite_Position", ("CAMERA_TIME", "EUCLIDEAN_DIM")),
        ("Satellite_Velocity", ("CAMERA_TIME", "EUCLIDEAN_DIM")),
        ("Azimuth_Motor_Position_SPICE", ("CAMERA_TIME",)),
        ("Azimuth_Motor_Position_SPICE_Int_Time_1", ("CAMERA_TIME",)),
        ("Azimuth_Motor_Position_SPICE_Int_Time_2", ("CAMERA_TIME",)),
    ],
)
def test_per_frame_geometry_is_written(l1b_product_dataset, variable, dims):
    """Each per-frame field is written on its product dimensions, covered at every DITL frame."""
    data_array = l1b_product_dataset[variable]
    assert data_array.dims == dims
    assert np.isfinite(data_array.to_numpy()).all(), f"{variable} contains NaN; SPICE coverage was expected"


def test_earth_sun_distance_is_written(l1b_product_dataset):
    """The granule Earth-Sun distance is written as a global attribute."""
    assert np.isfinite(float(l1b_product_dataset.attrs["Earth_Sun_Distance_AU"]))


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
    """Kept on-Earth pixels have finite geometry flagged 0; pixels the camera removed are NaN with bit 31.

    The flag variable keeps its uint32 dtype. DITL_3min mixes masked frames (about 8% kept) with an
    unmasked one.
    """
    flags = l1b_product_dataset["Geolocation_Quality_Flag"]
    assert flags.dims == ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    assert flags.dtype == np.uint32
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
    np.testing.assert_array_equal(flag_values[~kept], 1 << 31)
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
