"""Tests for the l1b algorithm"""
# Standard
from argparse import Namespace
# Installed
import pytest
import xarray as xr
# Local
from libera_utils.io.manifest import Manifest, ManifestType
from libera_cam.l1b import algorithm


@pytest.fixture
def generate_input_manifest(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data"""

    filenames = (test_data_path / "libera_rad_l1b_descriptor_20220909t000000_20220910t000000.h5",
                 test_data_path / "libera_rad_l1b_descriptor_20221010t000000_20221011t000000.h5")

    input_manifest = Manifest(ManifestType.INPUT, files=[], configuration={})

    input_manifest.add_file_to_manifest(filenames[0])
    input_manifest.add_file_to_manifest(filenames[1])

    input_manifest_file_path = input_manifest.write(outpath=tmp_path)

    return input_manifest_file_path


def test_algorithm(generate_input_manifest, monkeypatch, tmp_path):
    """Testing the algorithm to generate output manifests"""

    monkeypatch.setenv("PROCESSING_DROPBOX", str(tmp_path))
    algo_inputs = Namespace(manifest=str(generate_input_manifest))
    output_manifest_path = algorithm(algo_inputs)

    output_manifest_obj = Manifest.from_file(output_manifest_path)

    for file in output_manifest_obj.files:
        data_product = xr.open_dataset(file['filename'])
        print(data_product)
