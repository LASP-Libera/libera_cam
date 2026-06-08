"""Tests for the l1b algorithm"""

import os
from argparse import Namespace
from pathlib import Path

import pytest
import xarray as xr
from libera_utils.io.manifest import Manifest, ManifestType

from libera_cam.l1b import algorithm
from libera_cam.version import version as libera_cam_version


@pytest.fixture(scope="module")
def input_manifest_path(tmp_path_factory, test_data_path):
    """Generate a test input manifest from the DITL_short integration data."""
    ditl_data_path = test_data_path / "DITL_short"

    filenames = (
        ditl_data_path / "LIBERA_SPICE_AZROT-CK_V5-5-1_20280215T135304_20280215T142141_R26021234221.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20280215T000000_20280215T220000_R26006200700.bc",
        ditl_data_path / "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc",
        ditl_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp",
    )

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=filenames)
    return input_manifest.write(tmp_path_factory.mktemp("input_manifest"))


@pytest.fixture(scope="module")
def l1b_product_file_path(input_manifest_path, tmp_path_factory):
    """Run the L1B algorithm once per test module and return the output data file path."""
    tmp_path = tmp_path_factory.mktemp("l1b_product")
    os.environ["PROCESSING_PATH"] = str(tmp_path)
    try:
        output_manifest_path = algorithm(Namespace(manifest=str(input_manifest_path)))
    finally:
        del os.environ["PROCESSING_PATH"]

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
