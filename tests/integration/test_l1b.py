"""Tests for the l1b algorithm"""

import logging
from argparse import Namespace
from pathlib import Path

import pytest
import xarray as xr
from libera_utils.io.manifest import Manifest, ManifestType

from libera_cam.l1b import algorithm

# set logging to debug for the test
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture
def generate_input_manifest(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data"""
    ditl_data_path = test_data_path / "DITL_short"

    filenames = (
        ditl_data_path / "LIBERA_SPICE_AZROT-CK_V5-5-1_20280215T135304_20280215T142141_R26021234221.bc",
        ditl_data_path / "LIBERA_SPICE_JPSS-CK_V5-4-2_20280215T000000_20280215T220000_R26006200700.bc",
        ditl_data_path / "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc",
        ditl_data_path / "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp",
    )

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=filenames)

    input_manifest_file_path = input_manifest.write(tmp_path)

    return input_manifest_file_path


def test_algorithm(generate_input_manifest, monkeypatch, tmp_path):
    """Testing the algorithm to generate output manifests"""

    monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
    algo_inputs = Namespace(manifest=str(generate_input_manifest))
    output_manifest_path = algorithm(algo_inputs)

    output_manifest_obj = Manifest.from_file(output_manifest_path)

    for file in output_manifest_obj.files:
        if Path(file.filename).suffix == ".nc":
            data_product = xr.open_dataset(file.filename)
            print(data_product)
        if Path(file.filename).suffix == ".json":
            with open(file.filename) as f:
                metadata_content = f.read()
                print(metadata_content)
