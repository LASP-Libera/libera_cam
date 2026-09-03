import argparse
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import xarray as xr
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.manifest import Manifest

from libera_cam import l1b
from libera_cam.version import version as libera_cam_version

WFOV_L1A_FILENAME = "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"


def _mock_smart_open_file():
    mock_file = MagicMock()
    mock_file.__enter__ = Mock(return_value=mock_file)
    mock_file.__exit__ = Mock(return_value=False)
    return mock_file


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
        mock_manifest.configuration = {}
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
        mock_read.assert_called_with(mock_manifest)
        mock_process.assert_called_with(mock_l1a_data, mock_dynamic_kernel_sources, use_geo=True, jpss_only_mode=False)
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
        mock_file_info = MagicMock()
        mock_file_info.filename = "test_l1a.nc"
        az_file = MagicMock()
        az_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20280215T135304_20280215T142141_R26021234221.bc"
        jpss_spk = MagicMock()
        jpss_spk.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp"
        jpss_ck = MagicMock()
        jpss_ck.filename = "LIBERA_SPICE_JPSS-CK_V5-4-2_20280215T000000_20280215T220000_R26006200700.bc"
        mock_manifest = MagicMock()
        mock_manifest.files = [mock_file_info, az_file, jpss_spk, jpss_ck]
        mock_manifest.configuration = {}

        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.variables = ["var1"]
        mock_open_ds.return_value.load.return_value = mock_ds

        def _filename_from_path(path):
            mock_filename = MagicMock()
            if path.endswith(".nc"):
                mock_filename.data_product_id = DataProductIdentifier.l1a_icie_wfov_sci_decoded
            elif "AZROT-CK" in path:
                mock_filename.data_product_id = DataProductIdentifier.spice_az_ck
            elif "JPSS-SPK" in path:
                mock_filename.data_product_id = DataProductIdentifier.spice_jpss_spk
            elif "JPSS-CK" in path:
                mock_filename.data_product_id = DataProductIdentifier.spice_jpss_ck
            return mock_filename

        mock_filename_cls.from_file_path.side_effect = _filename_from_path

        with patch("libera_cam.l1b.smart_open", return_value=_mock_smart_open_file()):
            all_data, dynamic_kernel_sources = l1b.read_all_input_data(mock_manifest)

        assert "test_l1a.nc" in all_data
        assert all_data["test_l1a.nc"] == mock_ds
        assert dynamic_kernel_sources == [az_file.filename, jpss_spk.filename, jpss_ck.filename]
        mock_open_ds.return_value.load.assert_called_once()

    @patch("libera_cam.l1b.xr.open_dataset")
    @patch("libera_cam.l1b.LiberaDataProductFilename")
    def test_read_all_input_data_use_geo_false_skips_spice(self, mock_filename_cls, mock_open_ds):
        """use_geo false should skip SPICE files and return an empty kernel list."""
        nc_file = MagicMock()
        nc_file.filename = "test_l1a.nc"
        spice_file = MagicMock()
        spice_file.filename = "orbit.bc"

        mock_manifest = MagicMock()
        mock_manifest.files = [nc_file, spice_file]
        mock_manifest.configuration = {"use_geo": False}

        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.variables = ["var1"]
        mock_open_ds.return_value.load.return_value = mock_ds

        mock_filename = MagicMock()
        mock_filename.data_product_id = DataProductIdentifier.l1a_icie_wfov_sci_decoded
        mock_filename_cls.from_file_path.return_value = mock_filename

        with (
            patch("libera_cam.l1b.smart_open", return_value=_mock_smart_open_file()),
            self.assertLogs("libera_cam.l1b", level="WARNING") as log_context,
        ):
            all_data, dynamic_kernel_sources = l1b.read_all_input_data(mock_manifest)

        assert dynamic_kernel_sources == []
        assert "test_l1a.nc" in all_data
        assert any("use_geo is false: skipping SPICE kernel" in msg for msg in log_context.output)

    @patch("libera_cam.l1b.read_l1a_cam_data")
    @patch("libera_cam.l1b.convert_dn_to_radiance")
    @patch("libera_cam.l1b.add_azimuth_to_dataset")
    @patch("libera_cam.l1b.add_spacecraft_geometry_to_dataset")
    @patch("libera_cam.l1b.add_geolocation_to_dataset")
    def test_process_l1a_to_l1b_spice_mode(
        self, mock_geo, mock_spacecraft_geo, mock_azimuth, mock_convert, mock_read_l1a
    ):
        """Production mode: geolocation, spacecraft geometry and CK azimuth share one GeolocationKernelConfig."""
        mock_l1a_input = MagicMock(spec=xr.Dataset)
        all_input = {WFOV_L1A_FILENAME: mock_l1a_input}

        mock_lazy_ds = MagicMock(spec=xr.Dataset)
        mock_lazy_ds.image_data = MagicMock()
        mock_lazy_ds.integration_mask = MagicMock()
        mock_lazy_ds.chunk.return_value = mock_lazy_ds
        mock_read_l1a.return_value = mock_lazy_ds

        mock_radiance = MagicMock()
        mock_convert.return_value = mock_radiance

        mock_geo.return_value = mock_lazy_ds
        mock_spacecraft_geo.return_value = mock_lazy_ds
        mock_azimuth.return_value = mock_lazy_ds

        dynamic_kernel_sources = ["/tmp/spice/orbit.bc"]
        l1b.process_l1a_to_l1b(all_input, dynamic_kernel_sources, use_geo=True)

        mock_geo.assert_called_once()
        geo_config = mock_geo.call_args.args[1]
        # Every pixel is geolocated: no mask travels with the dataset.
        assert mock_geo.call_args.kwargs == {}
        assert mock_geo.call_args.args[0] is mock_lazy_ds
        assert geo_config.dynamic_kernel_sources == dynamic_kernel_sources

        # The spacecraft-level fields and the motor azimuth share the geolocation kernel config.
        mock_spacecraft_geo.assert_called_once_with(mock_lazy_ds, geo_config)
        mock_azimuth.assert_called_once_with(mock_lazy_ds, geo_config)

    @patch("libera_cam.l1b.read_l1a_cam_data")
    @patch("libera_cam.l1b.convert_dn_to_radiance")
    @patch("libera_cam.l1b.add_jpss_only_azimuth_to_dataset")
    @patch("libera_cam.l1b.add_spacecraft_geometry_to_dataset")
    @patch("libera_cam.l1b.add_jpss_only_geolocation_to_dataset")
    def test_process_l1a_to_l1b_jpss_only_mode(
        self, mock_jpss_geo, mock_spacecraft_geo, mock_jpss_azimuth, mock_convert, mock_read_l1a
    ):
        """jpss_only uses LIBERA_BASE per-pixel geolocation and the zero-degree reference Azimuth."""
        mock_l1a_input = MagicMock(spec=xr.Dataset)
        all_input = {WFOV_L1A_FILENAME: mock_l1a_input}

        mock_lazy_ds = MagicMock(spec=xr.Dataset)
        mock_lazy_ds.image_data = MagicMock()
        mock_lazy_ds.integration_mask = MagicMock()
        mock_lazy_ds.chunk.return_value = mock_lazy_ds
        mock_read_l1a.return_value = mock_lazy_ds

        mock_convert.return_value = MagicMock()
        mock_jpss_geo.return_value = mock_lazy_ds
        mock_spacecraft_geo.return_value = mock_lazy_ds
        mock_jpss_azimuth.return_value = mock_lazy_ds

        dynamic_kernel_sources = [
            "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp",
            "LIBERA_SPICE_JPSS-CK_V5-4-2_20280215T000000_20280215T220000_R26006200700.bc",
        ]
        result = l1b.process_l1a_to_l1b(all_input, dynamic_kernel_sources, jpss_only_mode=True)

        mock_jpss_geo.assert_called_once()
        assert mock_jpss_geo.call_args.kwargs == {}
        assert mock_jpss_geo.call_args.args[0] is mock_lazy_ds
        assert mock_jpss_geo.call_args.args[1].dynamic_kernel_sources == dynamic_kernel_sources
        # The spacecraft fields need no instrument frame, so jpss_only takes the same path.
        mock_spacecraft_geo.assert_called_once()
        mock_jpss_azimuth.assert_called_once_with(mock_lazy_ds)
        assert result is mock_lazy_ds

    @patch("libera_cam.l1b.read_l1a_cam_data")
    @patch("libera_cam.l1b.convert_dn_to_radiance")
    @patch("libera_cam.l1b.add_placeholder_azimuth_to_dataset")
    @patch("libera_cam.l1b.add_placeholder_spacecraft_geometry_to_dataset")
    @patch("libera_cam.l1b.add_placeholder_geolocation_to_dataset")
    def test_process_l1a_to_l1b_use_geo_false(
        self, mock_placeholder, mock_placeholder_spacecraft, mock_placeholder_azimuth, mock_convert, mock_read_l1a
    ):
        """use_geo false: the placeholder paths are called; SPICE path is not."""
        mock_l1a_input = MagicMock(spec=xr.Dataset)
        all_input = {WFOV_L1A_FILENAME: mock_l1a_input}

        mock_lazy_ds = MagicMock(spec=xr.Dataset)
        mock_lazy_ds.image_data = MagicMock()
        mock_lazy_ds.integration_mask = MagicMock()
        mock_lazy_ds.sizes = {"camera_time": 2}
        mock_lazy_ds.chunk.return_value = mock_lazy_ds
        mock_read_l1a.return_value = mock_lazy_ds

        mock_radiance = MagicMock()
        mock_convert.return_value = mock_radiance
        mock_placeholder.return_value = mock_lazy_ds
        mock_placeholder_spacecraft.return_value = mock_lazy_ds
        mock_placeholder_azimuth.return_value = mock_lazy_ds

        result = l1b.process_l1a_to_l1b(all_input, dynamic_kernel_sources=[], use_geo=False)

        mock_placeholder.assert_called_once_with(mock_lazy_ds)
        mock_placeholder_spacecraft.assert_called_once_with(mock_lazy_ds)
        mock_placeholder_azimuth.assert_called_once_with(mock_lazy_ds)
        assert result is mock_lazy_ds

    @patch("libera_cam.l1b.read_l1a_cam_data")
    @patch("libera_cam.l1b.convert_dn_to_radiance")
    def test_process_l1a_to_l1b_requires_kernel_sources_when_use_geo_true(self, mock_convert, mock_read_l1a):
        """use_geo true requires non-empty SPICE kernel sources."""
        mock_l1a_input = MagicMock(spec=xr.Dataset)
        all_input = {WFOV_L1A_FILENAME: mock_l1a_input}

        mock_lazy_ds = MagicMock(spec=xr.Dataset)
        mock_lazy_ds.image_data = MagicMock()
        mock_lazy_ds.integration_mask = MagicMock()
        mock_lazy_ds.chunk.return_value = mock_lazy_ds
        mock_read_l1a.return_value = mock_lazy_ds
        mock_convert.return_value = MagicMock()

        with pytest.raises(ValueError, match="SPICE kernel sources are required for geolocation when use_geo is True"):
            l1b.process_l1a_to_l1b(all_input, dynamic_kernel_sources=[], use_geo=True)

    @patch("libera_cam.l1b.write_libera_data_product")
    def test_write_data_product(self, mock_write_libera):
        """Test data product writing wrapper."""
        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.attrs = {}

        mock_filenames = (MagicMock(), MagicMock())
        mock_write_libera.return_value = mock_filenames

        result = l1b.write_data_product(mock_ds, "/tmp/out")

        assert result == mock_filenames
        assert mock_ds.attrs["algorithm_version"] == libera_cam_version()
        mock_write_libera.assert_called_once()


class TestReadAllInputDataSpiceKernels:
    @patch("libera_cam.l1b.xr.open_dataset")
    def test_read_all_input_data_jpss_only_mode_filters_kernels(self, mock_open_ds, caplog):
        """jpss_only collects only JPSS kernels and warns on motor kernels."""
        import logging

        nc_file = MagicMock()
        nc_file.filename = "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"
        az_file = MagicMock()
        az_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20280215T135304_20280215T142141_R26021234221.bc"
        jpss_spk = MagicMock()
        jpss_spk.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp"
        jpss_ck = MagicMock()
        jpss_ck.filename = "LIBERA_SPICE_JPSS-CK_V5-4-2_20280215T000000_20280215T220000_R26006200700.bc"

        mock_manifest = MagicMock()
        mock_manifest.files = [nc_file, az_file, jpss_spk, jpss_ck]
        mock_manifest.configuration = {"jpss_only": True}

        mock_ds = MagicMock(spec=xr.Dataset)
        mock_ds.variables = ["var1"]
        mock_open_ds.return_value.load.return_value = mock_ds

        with (
            patch("libera_cam.l1b.smart_open", return_value=_mock_smart_open_file()),
            caplog.at_level(logging.WARNING),
        ):
            _, dynamic_kernel_sources = l1b.read_all_input_data(mock_manifest)

        assert dynamic_kernel_sources == [jpss_spk.filename, jpss_ck.filename]
        assert "jpss_only mode: skipping SPICE file" in caplog.text

    @patch("libera_cam.l1b.xr.open_dataset")
    def test_read_all_input_data_production_collects_required_kernels(self, mock_open_ds):
        """Production mode collects AZROT + JPSS kernels in furnish order."""
        nc_file = MagicMock()
        nc_file.filename = "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"
        az_file = MagicMock()
        az_file.filename = "LIBERA_SPICE_AZROT-CK_V5-5-1_20280215T135304_20280215T142141_R26021234221.bc"
        jpss_spk = MagicMock()
        jpss_spk.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp"
        jpss_ck = MagicMock()
        jpss_ck.filename = "LIBERA_SPICE_JPSS-CK_V5-4-2_20280215T000000_20280215T220000_R26006200700.bc"

        mock_manifest = MagicMock()
        mock_manifest.files = [nc_file, jpss_ck, az_file, jpss_spk]
        mock_manifest.configuration = {}

        mock_open_ds.return_value.load.return_value = MagicMock(spec=xr.Dataset, variables=["var1"])

        with patch("libera_cam.l1b.smart_open", return_value=_mock_smart_open_file()):
            _, dynamic_kernel_sources = l1b.read_all_input_data(mock_manifest)

        assert dynamic_kernel_sources == [az_file.filename, jpss_spk.filename, jpss_ck.filename]

    @patch("libera_cam.l1b.xr.open_dataset")
    def test_read_all_input_data_duplicate_spice_raises(self, mock_open_ds):
        """Duplicate SPICE data product IDs in the manifest raise ValueError."""
        nc_file = MagicMock()
        nc_file.filename = "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"
        jpss_spk_a = MagicMock()
        jpss_spk_a.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp"
        jpss_spk_b = MagicMock()
        jpss_spk_b.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T235900_R26006200657.bsp"

        mock_manifest = MagicMock()
        mock_manifest.files = [nc_file, jpss_spk_a, jpss_spk_b]
        mock_manifest.configuration = {"jpss_only": True}

        mock_open_ds.return_value.load.return_value = MagicMock(spec=xr.Dataset, variables=["var1"])

        with patch("libera_cam.l1b.smart_open", return_value=_mock_smart_open_file()):
            with pytest.raises(ValueError, match="Duplicate SPICE data product"):
                l1b.read_all_input_data(mock_manifest)

    @patch("libera_cam.l1b.xr.open_dataset")
    def test_read_all_input_data_missing_required_spice_raises(self, mock_open_ds):
        """Missing required SPICE kernels raise ValueError."""
        nc_file = MagicMock()
        nc_file.filename = "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"
        jpss_spk = MagicMock()
        jpss_spk.filename = "LIBERA_SPICE_JPSS-SPK_V5-4-2_20280215T000000_20280215T220000_R26006200656.bsp"

        mock_manifest = MagicMock()
        mock_manifest.files = [nc_file, jpss_spk]
        mock_manifest.configuration = {"jpss_only": True}

        mock_open_ds.return_value.load.return_value = MagicMock(spec=xr.Dataset, variables=["var1"])

        with patch("libera_cam.l1b.smart_open", return_value=_mock_smart_open_file()):
            with pytest.raises(ValueError, match="missing required SPICE data products"):
                l1b.read_all_input_data(mock_manifest)


class TestAlgorithmUseGeoConfiguration:
    def test_algorithm_rejects_use_geo_false_and_jpss_only(self, tmp_path, monkeypatch):
        """Mutually exclusive manifest flags raise before processing."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        with (
            patch("libera_cam.l1b.Manifest.from_file") as mock_from_file,
            patch("libera_cam.l1b.read_all_input_data"),
        ):
            mock_from_file.return_value = Mock(
                files=[],
                configuration={"use_geo": False, "jpss_only": True},
            )
            with pytest.raises(ValueError, match="cannot both be enabled"):
                l1b.algorithm(argparse.Namespace(manifest="input.json"))

    def test_algorithm_use_geo_false_disables_geolocation(self, tmp_path, monkeypatch):
        """Explicit use_geo: false disables SPICE geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        output_manifest = Mock()
        with (
            patch("libera_cam.l1b.Manifest.from_file") as mock_from_file,
            patch("libera_cam.l1b.Manifest.output_manifest_from_input_manifest", return_value=output_manifest),
            patch("libera_cam.l1b.read_all_input_data") as mock_read,
            patch("libera_cam.l1b.process_l1a_to_l1b") as mock_process,
            patch("libera_cam.l1b.package_l1b_product"),
            patch("libera_cam.l1b.write_data_product") as mock_write,
        ):
            mock_from_file.return_value = Mock(
                files=[],
                configuration={"use_geo": False},
            )
            mock_read.return_value = ({}, [])
            mock_processed = MagicMock(spec=xr.Dataset)
            mock_process.return_value = mock_processed
            mock_write.return_value = (Mock(path=Path("output.nc")), Mock(path=Path("output.json")))
            output_manifest.write.return_value = tmp_path / "out_manifest.json"
            input_manifest = mock_from_file.return_value
            l1b.algorithm(argparse.Namespace(manifest="input.json"))
            mock_read.assert_called_once_with(input_manifest)
            assert mock_process.call_args.kwargs["use_geo"] is False
            assert mock_process.call_args.kwargs["jpss_only_mode"] is False

    def test_algorithm_passes_jpss_only_mode(self, tmp_path, monkeypatch):
        """jpss_only in configuration is forwarded to process_l1a_to_l1b."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        output_manifest = Mock()
        with (
            patch("libera_cam.l1b.Manifest.from_file") as mock_from_file,
            patch("libera_cam.l1b.Manifest.output_manifest_from_input_manifest", return_value=output_manifest),
            patch("libera_cam.l1b.read_all_input_data") as mock_read,
            patch("libera_cam.l1b.process_l1a_to_l1b") as mock_process,
            patch("libera_cam.l1b.package_l1b_product"),
            patch("libera_cam.l1b.write_data_product") as mock_write,
        ):
            mock_from_file.return_value = Mock(files=[], configuration={"jpss_only": True})
            mock_read.return_value = ({}, ["/tmp/jpss.bsp", "/tmp/jpss.bc"])
            mock_processed = MagicMock(spec=xr.Dataset)
            mock_process.return_value = mock_processed
            mock_write.return_value = (Mock(path=Path("output.nc")), Mock(path=Path("output.json")))
            output_manifest.write.return_value = tmp_path / "out_manifest.json"
            l1b.algorithm(argparse.Namespace(manifest="input.json"))
            assert mock_process.call_args.kwargs["jpss_only_mode"] is True

    def test_algorithm_omitted_use_geo_defaults_to_true(self, tmp_path, monkeypatch):
        """Omitting use_geo from configuration defaults to production geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        output_manifest = Mock()
        with (
            patch("libera_cam.l1b.Manifest.from_file") as mock_from_file,
            patch("libera_cam.l1b.Manifest.output_manifest_from_input_manifest", return_value=output_manifest),
            patch("libera_cam.l1b.read_all_input_data") as mock_read,
            patch("libera_cam.l1b.process_l1a_to_l1b") as mock_process,
            patch("libera_cam.l1b.package_l1b_product"),
            patch("libera_cam.l1b.write_data_product") as mock_write,
        ):
            mock_from_file.return_value = Mock(files=[], configuration={})
            mock_read.return_value = ({}, [])
            mock_processed = MagicMock(spec=xr.Dataset)
            mock_process.return_value = mock_processed
            mock_write.return_value = (Mock(path=Path("output.nc")), Mock(path=Path("output.json")))
            output_manifest.write.return_value = tmp_path / "out_manifest.json"
            input_manifest = mock_from_file.return_value
            l1b.algorithm(argparse.Namespace(manifest="input.json"))
            mock_read.assert_called_once_with(input_manifest)
            assert mock_process.call_args.kwargs["use_geo"] is True
            assert mock_process.call_args.kwargs["jpss_only_mode"] is False

    def test_algorithm_missing_processing_path(self, tmp_path, monkeypatch):
        """Test error when PROCESSING_PATH is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("libera_cam.l1b.Manifest.from_file") as mock_from_file,
        ):
            mock_from_file.return_value = Mock(files=[], configuration={})
            with pytest.raises(ValueError, match="PROCESSING_PATH environment variable is not set"):
                l1b.algorithm(argparse.Namespace(manifest="input.json"))
