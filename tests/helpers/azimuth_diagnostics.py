"""Diagnostics comparing the SPICE-derived motor azimuth with the FSW image-header azimuth.

Analysis tooling, not production code: the L1B product carries the CK-derived ``Azimuth`` and the
header value is dropped in packaging. Adapted from the comparison logging proposed in
libera_cam PR #16.
"""

import numpy as np


def angular_difference_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Shortest signed difference ``a - b`` on the circle, in degrees, in (-180, 180]."""
    return -((np.asarray(b_deg, dtype=np.float64) - np.asarray(a_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0)


def spice_vs_fsw_azimuth_stats(
    spice_az_deg: np.ndarray,
    fsw_az_rad: np.ndarray,
    fill_value: float = -999.0,
) -> dict[str, float]:
    """Statistics of SPICE azimuth (degrees) minus FSW header azimuth (radians), shortest arc.

    Parameters
    ----------
    spice_az_deg : np.ndarray
        Per-frame motor azimuth from the CK, degrees in [0, 360); ``fill_value`` where uncovered.
    fsw_az_rad : np.ndarray
        Per-frame FSW image-header azimuth in radians, same shape.
    fill_value : float, optional
        Value marking uncovered samples in either input; excluded from the comparison.

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
    fsw_rad = np.asarray(fsw_az_rad, dtype=np.float64)
    if spice.shape != fsw_rad.shape:
        raise ValueError(f"SPICE and FSW azimuth shapes differ: {spice.shape} vs {fsw_rad.shape}")

    valid = (spice != fill_value) & (fsw_rad != fill_value) & np.isfinite(spice) & np.isfinite(fsw_rad)
    if not valid.any():
        return {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}

    diff = angular_difference_deg(spice[valid], np.degrees(fsw_rad[valid]) % 360.0)
    return {
        "n": int(valid.sum()),
        "min": float(diff.min()),
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "std": float(diff.std()),
    }
