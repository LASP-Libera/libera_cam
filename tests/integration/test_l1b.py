"""Tests for the l1b algorithm"""

from argparse import Namespace
from pathlib import Path

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
