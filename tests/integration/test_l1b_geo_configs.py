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
        """Explicit use_geo false skips SPICE and writes product fill-value geolocation."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        nc_path = self._run_l1b(generate_input_manifest, {"use_geo": False}, tmp_path)

        with xr.open_dataset(nc_path, mask_and_scale=False) as dataset:
            assert np.all(dataset["Latitude"].values == np.float32(-999))
            assert np.all(dataset["Longitude"].values == np.float32(-999))
            assert np.all(dataset["Altitude"].values == np.float32(-9999))
            assert np.all(dataset["Azimuth"].values == np.float32(-999))
            assert np.any(np.isfinite(dataset["Radiance"].values))

    def test_jpss_only_runs_per_pixel_geolocation(self, generate_input_manifest, monkeypatch, tmp_path):
        """jpss_only mode writes per-pixel LIBERA_BASE geolocation and zero Azimuth."""
        monkeypatch.setenv("PROCESSING_PATH", str(tmp_path))
        nc_path = self._run_l1b(generate_input_manifest, {"jpss_only": True}, tmp_path)

        with xr.open_dataset(nc_path, mask_and_scale=False) as dataset:
            lat = dataset["Latitude"].values
            lon = dataset["Longitude"].values
            alt = dataset["Altitude"].values

            assert dataset["Altitude"].attrs.get("units") == "meters"
            assert np.any(np.isfinite(lat))
            assert np.any(np.isfinite(lon))

            valid = (lat != -999) & (lon != -999)
            invalid = ~valid
            assert np.any(valid)
            assert np.any(invalid)
            # Ellipsoid intersection height at the ground point is 0 m on the WGS84 surface.
            assert np.all(alt[valid] == 0.0)
            assert np.all(alt[invalid] == np.float32(-9999))

            # Per-pixel vectors should produce varying lat/lon across the FOV
            for t in range(lat.shape[0]):
                frame_valid = valid[t]
                n_valid = int(np.sum(frame_valid))
                if n_valid < 2:
                    continue
                frame_lats = lat[t, frame_valid]
                frame_lons = lon[t, frame_valid]
                assert np.std(frame_lats) > 0 or np.std(frame_lons) > 0, (
                    f"Frame {t}: expected per-pixel geolocation variation across {n_valid} valid pixels"
                )

            assert np.all(dataset["Azimuth"].values == 0)
            assert np.all(dataset["Solar_Zenith_Surface"].values == 0)
            assert np.all(dataset["Viewing_Zenith_Surface"].values == 0)
            assert np.all(dataset["Relative_Azimuth_Surface"].values == 0)
