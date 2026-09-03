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

    The DITL frames are ground-test data whose header azimuth does not track the CK (tens to
    hundreds of degrees apart), so the statistics are logged for analysis and only their
    coverage is asserted.
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
