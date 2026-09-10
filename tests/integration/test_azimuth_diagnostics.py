"""SPICE-versus-FSW azimuth comparison on the DITL data (diagnostic, not a product check)."""

import logging

import numpy as np
import pytest
import xarray as xr
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.geolocation import calculate_azimuth
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data
from tests.helpers.azimuth_diagnostics import spice_vs_fsw_azimuth_stats

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_spice_vs_fsw_azimuth_on_ditl(test_data_path, test_ditl_l1a_file_path):
    """Every DITL frame has both azimuths, so the comparison covers the whole granule.

    The DITL azimuth stage is rotating (about 0.5 deg/s), which is what makes the units
    testable: the header angle and the CK track each other to about 0.1 deg. Asserting that
    agreement pins the header field as degrees -- reintroducing a radians conversion moves the
    difference to over 100 deg and fails here.
    """
    ditl_dir = test_data_path / "DITL_3min"
    kernel_sources = sorted(p for p in ditl_dir.iterdir() if p.suffix in {".bc", ".bsp"})
    ds = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path, decode_times=True))

    with KernelManager() as km:
        km.load_libera_dynamic_kernels(kernel_sources, needs_naif_kernels=True, needs_static_kernels=True)
        spice_azimuth = calculate_azimuth(km, ds.camera_time.values)

    stats = spice_vs_fsw_azimuth_stats(spice_azimuth, ds["azimuth_angle"].values)
    logger.info("SPICE minus FSW azimuth on DITL_3min (degrees): %s", stats)

    assert stats["n"] == ds.sizes["camera_time"]
    assert all(np.isfinite(stats[key]) for key in ("min", "max", "mean", "std"))
    assert abs(stats["mean"]) < 1.0, f"header azimuth should track the CK in degrees, got {stats}"
    assert max(abs(stats["min"]), abs(stats["max"])) < 1.0

    # The increments must match 1:1: a radians/degrees error would show as a factor of 57.3.
    fsw = np.unwrap(np.radians(ds["azimuth_angle"].values)) * 180.0 / np.pi
    ck = np.unwrap(np.radians(spice_azimuth)) * 180.0 / np.pi
    d_fsw, d_ck = np.diff(fsw), np.diff(ck)
    moving = np.abs(d_ck) > 0.1
    if moving.any():
        ratio = d_fsw[moving] / d_ck[moving]
        np.testing.assert_allclose(ratio, 1.0, rtol=0.05)
