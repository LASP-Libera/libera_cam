"""SPICE-versus-FSW azimuth comparison on the DITL data (diagnostic, not a product check)."""

import logging

import numpy as np
import pytest
import xarray as xr
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.geolocation import calculate_azimuth
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data

logger = logging.getLogger(__name__)


def _angular_difference_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Shortest signed difference ``a - b`` on the circle, in degrees, in (-180, 180]."""
    return -((np.asarray(b_deg, dtype=np.float64) - np.asarray(a_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0)


def _spice_vs_fsw_azimuth_stats(
    spice_az_deg: np.ndarray,
    fsw_az_deg: np.ndarray,
) -> dict[str, float]:
    """Statistics of SPICE azimuth minus FSW header azimuth, both degrees, shortest arc.

    Both inputs are degrees. The FSW header records the encoder angle signed rather than
    wrapped to [0, 360), so it is wrapped here before differencing. On ``DITL_3min`` the two
    agree to 0.09-0.10 degrees, and their frame-to-frame increments match to within 0.4%.

    Parameters
    ----------
    spice_az_deg : np.ndarray
        Per-frame motor azimuth from the CK, degrees in [0, 360); NaN where uncovered.
    fsw_az_deg : np.ndarray
        Per-frame FSW image-header azimuth in degrees, signed, same shape. NaN samples in either
        input are excluded from the comparison.

    Returns
    -------
    dict[str, float]
        ``n`` (compared samples), ``min``, ``max``, ``mean`` and ``std`` of the difference in
        degrees. The statistics are NaN when ``n`` is 0.

    Raises
    ------
    ValueError
        If the two inputs differ in shape.
    """
    spice = np.asarray(spice_az_deg, dtype=np.float64)
    fsw = np.asarray(fsw_az_deg, dtype=np.float64)
    if spice.shape != fsw.shape:
        raise ValueError(f"SPICE and FSW azimuth shapes differ: {spice.shape} vs {fsw.shape}")

    valid = np.isfinite(spice) & np.isfinite(fsw)
    if not valid.any():
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}

    diff = _angular_difference_deg(spice[valid], fsw[valid] % 360.0)
    return {
        "n": int(valid.sum()),
        "min": float(diff.min()),
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "std": float(diff.std()),
    }


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

    stats = _spice_vs_fsw_azimuth_stats(spice_azimuth, ds["azimuth_angle"].values)
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
