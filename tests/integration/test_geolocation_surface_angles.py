"""Integration tests for per-pixel surface geometry angle helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.geolocation import calculate_pixel_surface_geometry_angles

pytestmark = pytest.mark.integration

_LIBERA_RAD_KERNEL_DIR = (
    Path(__file__).resolve().parents[2].parent / "libera_rad" / "tests" / "test_data" / "l1b_integration_data"
)


def _jpss_kernel_sources(kernel_dir: Path) -> list[str]:
    return sorted(
        str(kernel_dir / name)
        for name in (
            "LIBERA_SPICE_JPSS-CK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bc",
            "LIBERA_SPICE_JPSS-SPK_V5-4-2_20251120T000000_20251120T235900_R26016205551.bsp",
        )
        if (kernel_dir / name).is_file()
    )


@pytest.mark.skipif(not _LIBERA_RAD_KERNEL_DIR.is_dir(), reason="libera_rad JPSS test kernels not available")
def test_calculate_pixel_surface_geometry_angles_jpss_kernels():
    """Surface angles at a subsatellite-like point should be physically plausible."""
    sources = _jpss_kernel_sources(_LIBERA_RAD_KERNEL_DIR)
    if len(sources) != 2:
        pytest.skip("libera_rad JPSS test kernel files not available")

    times = pd.date_range("2025-11-20 18:00:00", periods=2, freq="1s").to_numpy(dtype="datetime64[ns]")
    lat = np.zeros((2, 1, 1), dtype=np.float64)
    lon = np.zeros((2, 1, 1), dtype=np.float64)
    alt = np.zeros((2, 1, 1), dtype=np.float64)

    with KernelManager() as km:
        km.load_libera_dynamic_kernels(sources, needs_naif_kernels=True, needs_static_kernels=True)
        angles = calculate_pixel_surface_geometry_angles(
            km,
            times,
            lat,
            lon,
            alt,
            spacecraft_body="LIBERA_BASE",
        )

    sza = angles["solar_zenith"].ravel()
    vza = angles["viewing_zenith"].ravel()
    raa = angles["relative_azimuth"].ravel()
    saa = angles["solar_azimuth"].ravel()
    vaa = angles["viewing_azimuth"].ravel()

    assert np.all((sza >= 0) & (sza <= 180))
    assert np.all((vza >= 0) & (vza <= 180))
    assert np.all((raa >= 0) & (raa < 360))
    assert np.all((saa >= 0) & (saa < 360))
    assert np.all((vaa >= 0) & (vaa < 360))
    assert np.all(sza != -999)
