"""Tests for the L1B orchestration module"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.manifest import Manifest

from libera_cam import l1b


@pytest.fixture
def mock_manifest():
    """Create a mock input manifest"""
    manifest = MagicMock(spec=Manifest)
    manifest.files = []
    # Mocking file entries
    file_info = MagicMock()
    file_info.filename = "s3://bucket/test_l1a.nc"
    manifest.files.append(file_info)

    spice_info = MagicMock()
    spice_info.filename = "s3://bucket/kernel.bsp"
    manifest.files.append(spice_info)

    return manifest


@patch("libera_cam.l1b.Manifest")
@patch("libera_cam.l1b.read_all_input_data")
@patch("libera_cam.l1b.process_l1a_to_l1b")
@patch("libera_cam.l1b.write_data_product")
def test_algorithm(mock_write, mock_process, mock_read, mock_manifest_cls, mock_manifest):
    """
    Test the main algorithm orchestration function.
    Verifies:
    1. Input manifest is read.
    2. Data is read (Step 2).
    3. Processing is called (Step 3).
    4. Output is written (Step 4).
    5. Output manifest is created and written (Step 5-8).
    """
    # Setup Mocks
    # 1. Manifest.from_file returns mock_manifest
    mock_manifest_cls.from_file.return_value = mock_manifest

    # 2. Setup the output manifest mock chain
    # The code calls Manifest.output_manifest_from_input_manifest(input_manifest)
    mock_out_man = MagicMock(spec=Manifest)
    mock_manifest_cls.output_manifest_from_input_manifest.return_value = mock_out_man

    # Mock return values for steps
    mock_l1a_data = {"test": "data"}
    mock_spice_dir = Path("/tmp/spice")
    mock_read.return_value = (mock_l1a_data, mock_spice_dir)

    mock_processed_ds = MagicMock(spec=xr.Dataset)
    mock_process.return_value = mock_processed_ds

    mock_output_data_filename = MagicMock()
    mock_output_data_filename.path = Path("output.nc")
    mock_output_ummg_filename = Path("output.json")
    mock_write.return_value = (mock_output_data_filename, mock_output_ummg_filename)

    # Call the algorithm
    args = argparse.Namespace(manifest="input.json")
    with patch.dict("os.environ", {"PROCESSING_PATH": "/tmp/dropbox"}):
        _ = l1b.algorithm(args)

    # Verifications
    mock_manifest_cls.from_file.assert_called_once_with("input.json")

    mock_read.assert_called_once_with(mock_manifest)

    mock_process.assert_called_once_with(mock_l1a_data, mock_spice_dir)

    mock_write.assert_called_once_with(mock_processed_ds, "/tmp/dropbox")

    # Check Output Manifest interactions
    mock_out_man.add_files.assert_any_call(Path("output.nc"))
    mock_out_man.add_files.assert_any_call(Path("output.json"))
    mock_out_man.write.assert_called_once_with("/tmp/dropbox")


@patch("libera_cam.l1b.smart_open")
@patch("libera_cam.l1b.smart_copy_file")
@patch("libera_cam.l1b.xr.open_dataset")
def test_read_all_input_data(mock_xr_open, mock_copy, mock_smart_open, mock_manifest):
    """
    Test read_all_input_data.
    Verifies:
    1. SPICE files (.bsp/.bc) are copied locally.
    2. NetCDF files (.nc) are opened via xarray.
    """
    # Setup
    mock_file_nc = MagicMock()
    # Use a filename that matches the regex in LiberaDataProductFilename
    mock_file_nc.filename = "LIBERA_L1A_WFOV-SCI-PDS_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"

    mock_file_spice = MagicMock()
    mock_file_spice.filename = "kernel.bsp"

    mock_manifest.files = [mock_file_nc, mock_file_spice]

    # Run
    data, spice_dir = l1b.read_all_input_data(mock_manifest)

    # Verify SPICE Copy
    assert spice_dir.name == "spice_files"
    # Ensure smart_copy was called for the .bsp file
    mock_copy.assert_called_once()
    args, _ = mock_copy.call_args
    assert args[0] == "kernel.bsp"
    assert str(args[1]).endswith("kernel.bsp")

    # Verify NetCDF Open
    # Use the actual mock object returned by __enter__ for comparison
    mock_file_handle = mock_smart_open.return_value.__enter__.return_value

    # We need to simulate the loop logic.
    # The function calls smart_open(filename) -> xr.open_dataset(file_handle)

    mock_smart_open.assert_called_once_with(mock_file_nc.filename)
    mock_xr_open.assert_called_once_with(mock_file_handle)


@patch("libera_cam.l1b.read_l1a_cam_data")
@patch("libera_cam.l1b.convert_dn_to_radiance")
@patch("libera_cam.l1b.add_geolocation_to_dataset")
@patch("libera_cam.l1b.package_l1b_product")
def test_process_l1a_to_l1b(mock_package, mock_geo, mock_rad, mock_read_l1a):
    """
    Test the science processing pipeline.
    Verifies the chain of calls:
    Read -> Calibration -> Geolocation -> Packaging
    """
    # Inputs
    mock_ds = MagicMock(spec=xr.Dataset)
    input_data = {DataProductIdentifier.l1a_icie_wfov_sci_decoded: mock_ds}
    spice_dir = Path("/tmp/spice")

    # Mocking Intermediate Returns
    mock_cam_ds = MagicMock(spec=xr.Dataset)
    mock_cam_ds.image_data = "image_data"
    mock_cam_ds.integration_mask = "int_mask"
    mock_cam_ds.valid_pixel_mask = "pixel_mask"
    # Mock chunking returning self (or modified self)
    mock_cam_ds.chunk.return_value = mock_cam_ds

    mock_read_l1a.return_value = mock_cam_ds

    mock_calibrated_data = MagicMock()
    mock_calibrated_data.data = "radiance_values"
    mock_rad.return_value = mock_calibrated_data

    mock_geo_ds = MagicMock(spec=xr.Dataset)
    mock_geo.return_value = mock_geo_ds

    mock_final_ds = MagicMock(spec=xr.Dataset)
    mock_package.return_value = mock_final_ds

    # Run
    result = l1b.process_l1a_to_l1b(input_data, spice_dir)

    # Verify Call Chain
    mock_read_l1a.assert_called_once_with(mock_ds)
    mock_rad.assert_called_once_with("image_data", "int_mask")

    # Verify Geolocation call
    mock_geo.assert_called_once()
    args, kwargs = mock_geo.call_args
    assert args[0] == mock_cam_ds  # Passed the dataset
    # Pixel mask is passed as kwarg
    assert kwargs["pixel_mask"] == "pixel_mask"

    mock_package.assert_called_once_with(mock_geo_ds)
    assert result == mock_final_ds


@patch("libera_cam.l1b.write_libera_data_product")
def test_write_data_product(mock_write_libera):
    """Test writing data product wrapper."""
    mock_ds = MagicMock(spec=xr.Dataset)
    output_path = "/tmp/out"

    l1b.write_data_product(mock_ds, output_path)

    mock_write_libera.assert_called_once()
    call_kwargs = mock_write_libera.call_args[1]

    assert call_kwargs["data"] == mock_ds
    assert call_kwargs["output_path"] == output_path
    assert str(call_kwargs["data_product_definition"]).endswith("L1B_CAM_product_definition.yml")
