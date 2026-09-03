import numpy as np
import pytest

from tests.helpers.azimuth_diagnostics import angular_difference_deg, spice_vs_fsw_azimuth_stats


def test_angular_difference_is_shortest_signed_arc():
    a = np.array([10.0, 350.0, 180.0, 0.0])
    b = np.array([350.0, 10.0, 0.0, 180.0])
    np.testing.assert_allclose(angular_difference_deg(a, b), [20.0, -20.0, 180.0, 180.0])


def test_stats_convert_radians_and_exclude_fill():
    spice_deg = np.array([10.0, 90.0, -999.0, 45.0])
    fsw_rad = np.radians(np.array([350.0, 80.0, 30.0, -999.0]))
    fsw_rad[3] = -999.0

    stats = spice_vs_fsw_azimuth_stats(spice_deg, fsw_rad)

    assert stats["n"] == 2
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(20.0)
    assert stats["mean"] == pytest.approx(15.0)
    assert stats["std"] == pytest.approx(5.0)


def test_stats_with_no_valid_samples_are_nan():
    stats = spice_vs_fsw_azimuth_stats(np.array([-999.0, np.nan]), np.array([0.1, 0.2]))
    assert stats["n"] == 0
    assert all(np.isnan(stats[key]) for key in ("min", "max", "mean", "std"))


def test_stats_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        spice_vs_fsw_azimuth_stats(np.zeros(3), np.zeros(2))
