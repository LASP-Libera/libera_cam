import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.manifest import Manifest

from libera_cam import l1b


class TestL1b(unittest.TestCase):
    @patch("libera_cam.l1b.Manifest")
    @patch("libera_cam.l1b.read_all_input_data")
    @patch("libera_cam.l1b.process_l1a_to_l1b")
    @patch("libera_cam.l1b.package_l1b_product")
    @patch("libera_cam.l1b.write_data_product")
    def test_algorithm(self, mock_write, mock_package, mock_process, mock_read, mock_manifest_cls):
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
        mock_manifest = MagicMock(spec=Manifest)
        mock_manifest.files = []
        mock_manifest.configuration = {}  # without a no_geo key -> production mode
        mock_manifest_cls.from_file.return_value = mock_manifest

        # Setup the output manifest mock chain
        mock_out_man = MagicMock(spec=Manifest)
        mock_out_man.configuration = MagicMock()
        mock_manifest_cls.output_manifest_from_input_manifest.return_value = mock_out_man

        # Mock return values for steps
        mock_l1a_data = {"test": "data"}
        mock_dynamic_kernel_sources = ["/tmp/spice/orbit.bc"]
        mock_read.return_value = (mock_l1a_data, mock_dynamic_kernel_sources)

        mock_processed_ds = MagicMock(spec=xr.Dataset)
        mock_process.return_value = mock_processed_ds

        # Correctly mock LiberaDataProductFilename objects which have a .path attribute
        mock_output_data_filename = MagicMock()
        mock_output_data_filename.path = Path("output.nc")
        mock_output_ummg_filename = MagicMock()
        mock_output_ummg_filename.path = Path("output.json")

        # write_data_product returns a tuple of filenames
        mock_write.return_value = (mock_output_data_filename, mock_output_ummg_filename)

        # Call the algorithm
        args = argparse.Namespace(manifest="input.json")
        with patch.dict("os.environ", {"PROCESSING_PATH": "/tmp/dropbox"}):
            _ = l1b.algorithm(args)

        # Verification
        mock_manifest_cls.from_file.assert_called_with("input.json")
        mock_read.assert_called_with(mock_manifest, no_geo_mode=False)
        mock_process.assert_called_with(mock_l1a_data, mock_dynamic_kernel_sources, no_geo_mode=False)
        mock_package.assert_called_once_with(mock_processed_ds)
        mock_write.assert_called_with(mock_package.return_value, "/tmp/dropbox")

        # Verify input configuration is propagated to the output manifest
        mock_out_man.configuration.update.assert_called_once_with({})

        # Verify manifest file addition (both files)
        assert mock_out_man.add_files.call_count == 2
        mock_out_man.add_files.assert_any_call(Path("output.nc"))
        mock_out_man.add_files.assert_any_call(Path("output.json"))

        mock_out_man.write.assert_called_with("/tmp/dropbox")

    @patch("libera_cam.l1b.Manifest")
    @patch("libera_cam.l1b.xr.open_dataset")
    @patch("libera_cam.l1b.LiberaDataProductFilename")
    def test_read_all_input_data(self, mock_filename_cls, mock_open_ds, mock_manifest_cls):
        """Test manifest file reading and dataset loading."""
        # Setup Manifest mock with one NetCDF file
        mock_file_info = MagicMock()
        mock_file_info.filename = "test_l1a.nc"
        mock_manifest = MagicMock()
        mock_manifest.files = [mock_file_info]

        # Setup Dataset mock
        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.variables = ["var1"]
        mock_open_ds.return_value = mock_ds

        # Setup Filename mock
        mock_filename = MagicMock()
        mock_filename.data_product_id = DataProductIdentifier.l1a_icie_wfov_sci_decoded
        mock_filename_cls.return_value = mock_filename

        all_data, dynamic_kernel_sources = l1b.read_all_input_data(mock_manifest)

        # Verify
        assert "WFOV-SCI-DECODED" in all_data
        assert all_data["WFOV-SCI-DECODED"] == mock_ds
        assert dynamic_kernel_sources == []
        # Verify that .load() was NOT called (maintaining laziness)
        mock_ds.load.assert_not_called()

    @patch("libera_cam.l1b.xr.open_dataset")
    @patch("libera_cam.l1b.LiberaDataProductFilename")
    def test_read_all_input_data_ground_data_mode(self, mock_filename_cls, mock_open_ds):
        """SPICE files are silently skipped and dynamic_kernel_sources is None in no_geo_mode."""
        nc_file = MagicMock()
        nc_file.filename = "test_l1a.nc"
        spice_file = MagicMock()
        spice_file.filename = "orbit.bc"

        mock_manifest = MagicMock()
        mock_manifest.files = [nc_file, spice_file]

        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.variables = ["var1"]
        mock_open_ds.return_value = mock_ds

        mock_filename = MagicMock()
        mock_filename.data_product_id = DataProductIdentifier.l1a_icie_wfov_sci_decoded
        mock_filename_cls.return_value = mock_filename

        all_data, dynamic_kernel_sources = l1b.read_all_input_data(mock_manifest, no_geo_mode=True)

        assert dynamic_kernel_sources is None
        assert "WFOV-SCI-DECODED" in all_data

    @patch("libera_cam.l1b.read_l1a_cam_data")
    @patch("libera_cam.l1b.convert_dn_to_radiance")
    @patch("libera_cam.l1b.add_geolocation_to_dataset")
    def test_process_l1a_to_l1b_spice_mode(self, mock_geo, mock_convert, mock_read_l1a):
        """Production mode: add_geolocation_to_dataset is called with a GeolocationKernelConfig."""
        mock_l1a_input = MagicMock(spec=xr.Dataset)
        all_input = {DataProductIdentifier.l1a_icie_wfov_sci_decoded: mock_l1a_input}

        mock_lazy_ds = MagicMock(spec=xr.Dataset)
        mock_lazy_ds.image_data = MagicMock()
        mock_lazy_ds.integration_mask = MagicMock()
        mock_lazy_ds.valid_pixel_mask = MagicMock()
        mock_lazy_ds.chunk.return_value = mock_lazy_ds
        mock_read_l1a.return_value = mock_lazy_ds

        mock_radiance = MagicMock()
        mock_convert.return_value = mock_radiance

        mock_geo.return_value = mock_lazy_ds

        dynamic_kernel_sources = ["/tmp/spice/orbit.bc"]
        l1b.process_l1a_to_l1b(all_input, dynamic_kernel_sources, no_geo_mode=False)

        mock_geo.assert_called_once()
        call_kwargs = mock_geo.call_args
        # Second positional arg is the GeolocationKernelConfig
        assert call_kwargs.args[1].dynamic_kernel_sources == dynamic_kernel_sources
        # Third keyword arg is pixel_mask
        assert call_kwargs.kwargs["pixel_mask"] is mock_lazy_ds.valid_pixel_mask

    @patch("libera_cam.l1b.read_l1a_cam_data")
    @patch("libera_cam.l1b.convert_dn_to_radiance")
    @patch("libera_cam.l1b.add_placeholder_geolocation_to_dataset")
    def test_process_l1a_to_l1b_no_geo_mode(self, mock_placeholder, mock_convert, mock_read_l1a):
        """No geolocation mode: add_placeholder_geolocation_to_dataset is called; SPICE path is not."""
        mock_l1a_input = MagicMock(spec=xr.Dataset)
        all_input = {DataProductIdentifier.l1a_icie_wfov_sci_decoded: mock_l1a_input}

        mock_lazy_ds = MagicMock(spec=xr.Dataset)
        mock_lazy_ds.image_data = MagicMock()
        mock_lazy_ds.integration_mask = MagicMock()
        mock_lazy_ds.chunk.return_value = mock_lazy_ds
        mock_read_l1a.return_value = mock_lazy_ds

        mock_radiance = MagicMock()
        mock_convert.return_value = mock_radiance
        mock_placeholder.return_value = mock_lazy_ds

        result = l1b.process_l1a_to_l1b(all_input, dynamic_kernel_sources=None, no_geo_mode=True)

        mock_placeholder.assert_called_once_with(mock_lazy_ds)
        assert result is mock_lazy_ds

    @patch("libera_cam.l1b.write_libera_data_product")
    @patch("libera_cam.l1b.resources.files")
    def test_write_data_product(self, mock_resources, mock_write_libera):
        """Test data product writing wrapper."""
        mock_ds = MagicMock(spec=xr.Dataset)
        mock_resources.return_value.joinpath.return_value = "product_def.yml"

        mock_filenames = (MagicMock(), MagicMock())
        mock_write_libera.return_value = mock_filenames

        result = l1b.write_data_product(mock_ds, "/tmp/out")

        assert result == mock_filenames
        mock_write_libera.assert_called_once()
