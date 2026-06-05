"""Integration tests for L1B manifest configuration.use_geo."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import spiceypy as sp
import xarray as xr
from libera_utils.io.manifest import Manifest

from libera_cam.l1b import algorithm


class TestL1bManifestUseGeoConfiguration:
    """Each manifest configuration case is a separate integration test."""

    @pytest.fixture(autouse=True)
    def clear_spice_state(self):
        try:
            sp.kclear()
        except Exception:
            pass
        yield
        try:
            sp.kclear()
        except Exception:
            pass

    def _output_nc_path(self, output_manifest: Manifest) -> str:
        for file in output_manifest.files:
            if Path(file.filename).suffix == ".nc":
                return file.filename
        raise AssertionError("No NetCDF data product in output manifest")

    def _run_l1b(self, generate_input_manifest, configuration: dict, tmp_path):
        manifest_path = generate_input_manifest(configuration)
        input_manifest = Manifest.from_file(manifest_path)
        output_manifest = Manifest.from_file(algorithm(Namespace(manifest=str(manifest_path))))
        for key, value in input_manifest.configuration.items():
            assert output_manifest.configuration[key] == value
        return self._output_nc_path(output_manifest)

    def test_use_geo_absent_runs_spice_geolocation(self, generate_input_manifest, monkeypatch, tmp_path):
        """Omitting use_geo from configuration runs SPICE geolocation (production default)."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        nc_path = self._run_l1b(generate_input_manifest, {}, tmp_path)

        with xr.open_dataset(nc_path, mask_and_scale=False) as dataset:
            assert "Latitude" in dataset
            assert "Longitude" in dataset
            assert "Altitude" in dataset
            assert "Radiance" in dataset
            assert np.any(np.isfinite(dataset["Radiance"].values))

    def test_use_geo_true_runs_spice_geolocation(self, generate_input_manifest, monkeypatch, tmp_path):
        """Explicit use_geo true runs SPICE geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        nc_path = self._run_l1b(generate_input_manifest, {"use_geo": True}, tmp_path)

        with xr.open_dataset(nc_path, mask_and_scale=False) as dataset:
            assert "Latitude" in dataset
            assert "Longitude" in dataset
            assert "Altitude" in dataset
            assert np.any(np.isfinite(dataset["Radiance"].values))

    def test_use_geo_false_writes_placeholder_geolocation(self, generate_input_manifest, monkeypatch, tmp_path):
        """Explicit use_geo false skips SPICE and writes NaN placeholder geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        nc_path = self._run_l1b(generate_input_manifest, {"use_geo": False}, tmp_path)

        with xr.open_dataset(nc_path, mask_and_scale=False) as dataset:
            assert np.all(np.isnan(dataset["Latitude"].values))
            assert np.all(np.isnan(dataset["Longitude"].values))
            assert np.all(np.isnan(dataset["Altitude"].values))
            assert np.any(np.isfinite(dataset["Radiance"].values))
