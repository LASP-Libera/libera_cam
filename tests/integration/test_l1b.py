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
        ditl_data_path / "LIBERA_L1A_WFOV-SCI-DECODED_V5-8-4_20280212T080001_20280212T080342_R26157025734.nc",
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
    ],
)
def test_spacecraft_geometry_is_computed_per_frame(l1b_product_dataset, variable, low, high):
    """Each spacecraft-level field is one covered value per camera frame, inside its valid range."""
    data_array = l1b_product_dataset[variable]
    assert data_array.dims == ("CAMERA_TIME",)

    values = data_array.to_numpy()
    assert not np.any(values <= -999.0), f"{variable} contains _FillValue; SPICE coverage was expected"
    assert values.min() >= low
    assert values.max() <= high


def test_subsatellite_colatitude_complements_latitude(l1b_product_dataset):
    """Colatitude is the geodetic complement of latitude, so the two must stay consistent."""
    latitude = l1b_product_dataset["Subsatellite_Latitude"].to_numpy()
    colatitude = l1b_product_dataset["Subsatellite_Colatitude"].to_numpy()
    np.testing.assert_allclose(colatitude, 90.0 - latitude, atol=1e-4)
