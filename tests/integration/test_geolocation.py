"""Integration tests for the lazy per-pixel geolocation on the DITL_3min data."""

from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from curryer import spicetime
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.constants import GROUND_CAL_PIXEL_MAPPING
from libera_cam.geolocation import (
    _FIELD_VARIABLES,
    GeolocationKernelConfig,
    add_geolocation_to_dataset,
    geolocate_frame,
)
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data

pytestmark = pytest.mark.integration


def _ditl_dynamic_kernel_sources(test_data_path: Path) -> list[Path]:
    d = test_data_path / "DITL_3min"
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"})


def test_add_geolocation_to_dataset_reproduces_geolocate_frame(test_data_path, test_ditl_l1a_file_path, monkeypatch):
    """The lazy path splits frames into geometry tasks and writes exactly what geolocate_frame returns."""
    monkeypatch.setenv("LIBERA_CAM_GEO_CHUNK_SIZE", "2")
    ds = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path)).isel(camera_time=slice(0, 3))
    sources = _ditl_dynamic_kernel_sources(test_data_path)
    config = GeolocationKernelConfig(dynamic_kernel_sources=sources)

    ds_geo = add_geolocation_to_dataset(ds, config)

    variables = [variable for variable, _ in _FIELD_VARIABLES.values()]
    for variable, dtype in _FIELD_VARIABLES.values():
        assert isinstance(ds_geo[variable].data, da.Array), variable
        assert ds_geo[variable].dims == ("camera_time", "y", "x")
        assert ds_geo[variable].dtype == dtype, variable
        assert ds_geo[variable].data.chunks[0] == (2, 1), variable

    # The last frame sits alone in the second task.
    computed = ds_geo[variables].isel(camera_time=2).compute(scheduler="synchronous")
    with KernelManager() as km:
        km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        ugps = np.asarray(spicetime.adapt(pd.DatetimeIndex(ds.camera_time.values[2:3]), "iso"))
        vectors = np.load(GROUND_CAL_PIXEL_MAPPING, mmap_mode="r").reshape(-1, 3)
        expected = geolocate_frame(ugps, None, "LIBERA_WFOV_CAM", vectors)

    for name, (variable, _) in _FIELD_VARIABLES.items():
        np.testing.assert_array_equal(computed[variable].values, getattr(expected, name), err_msg=variable)
    on_earth = computed["Geolocation_Quality_Flag"].values == 0
    assert 0.70 < on_earth.mean() < 0.85


def test_add_geolocation_rejects_a_granule_the_kernels_do_not_cover(test_data_path, test_ditl_l1a_file_path):
    """Frames a day past the kernels: no frame is covered, so the granule is rejected before any task runs."""
    ds = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path)).isel(camera_time=slice(0, 2))
    ds = ds.assign_coords(camera_time=ds.camera_time + np.timedelta64(1, "D"))
    config = GeolocationKernelConfig(dynamic_kernel_sources=_ditl_dynamic_kernel_sources(test_data_path))

    with pytest.raises(RuntimeError, match="cover none of the 2 camera frame"):
        add_geolocation_to_dataset(ds, config)


def test_dynamic_kernel_sequence_materializes_into_cache(monkeypatch, tmp_path, test_data_path):
    """Explicit sequence mode should materialize kernel basenames into the user cache via KernelFileCache."""
    test_kernel_dir = test_data_path / "DITL_3min"
    kernel_files = sorted(
        [p for p in test_kernel_dir.iterdir() if p.is_file() and p.suffix in {".bc", ".bsp"}],
        key=lambda p: p.name,
    )
    assert kernel_files, f"No dynamic kernels found under {test_kernel_dir}"
    sources = kernel_files[:2]

    monkeypatch.setattr(spice_utils.caching, "get_local_cache_dir", lambda: tmp_path)

    km = KernelManager(cache_timeout_days=7)
    km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)

    for src in sources:
        assert (tmp_path / src.name).is_file(), f"Expected cached kernel missing: {src.name}"
