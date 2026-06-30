"""Unit tests for SPICE motor azimuth and FSW comparison helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from libera_cam.geolocation import (
    GeolocationKernelConfig,
    _angular_diff_deg,
    apply_spice_azimuth_to_dataset,
    calculate_azimuth_for_timestamps,
    log_spice_vs_fsw_azimuth_differences,
)


def test_angular_diff_deg_wraps_shortest_arc():
    diff = _angular_diff_deg(np.array([10.0, 350.0]), np.array([350.0, 10.0]))
    np.testing.assert_allclose(diff, [20.0, -20.0], rtol=0, atol=1e-12)


@patch("libera_cam.geolocation.sp.pxform")
@patch("libera_cam.geolocation.sp.m2eul")
@patch("libera_cam.geolocation.spicetime.adapt", return_value=np.array([1.0, 2.0]))
def test_calculate_azimuth_for_timestamps(mock_adapt, mock_m2eul, mock_pxform):
    mock_pxform.return_value = np.eye(3)
    mock_m2eul.return_value = (0.0, 0.0, np.pi / 2.0)
    km = MagicMock()

    timestamps = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:01"], dtype="datetime64[ns]")
    az = calculate_azimuth_for_timestamps(km, timestamps)

    km.ensure_known_kernels_are_furnished.assert_called_once()
    assert az.shape == (2,)
    assert az.dtype == np.float32
    assert az[0] == pytest.approx(90.0)
    assert az[1] == pytest.approx(90.0)


def test_log_spice_vs_fsw_azimuth_differences(caplog):
    spice_deg = np.array([90.0, 100.0, -999.0], dtype=np.float32)
    fsw_rad = np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64)

    with caplog.at_level("INFO"):
        log_spice_vs_fsw_azimuth_differences(spice_deg, fsw_rad)

    assert any("SPICE vs FSW azimuth comparison" in record.message for record in caplog.records)
    assert any("diff_deg min=" in record.message for record in caplog.records)


@patch("libera_cam.geolocation.log_spice_vs_fsw_azimuth_differences")
@patch("libera_cam.geolocation.calculate_azimuth_for_timestamps")
@patch("libera_cam.geolocation.KernelManager")
@patch("libera_cam.geolocation.prefetch_kernels")
def test_apply_spice_azimuth_logs_fsw_before_overwrite(mock_prefetch, mock_km_cls, mock_calc_az, mock_log_diff):
    mock_km = MagicMock()
    mock_km_cls.return_value = mock_km
    mock_km.__enter__.return_value = mock_km
    mock_km.__exit__.return_value = False
    mock_calc_az.return_value = np.array([90.0, 180.0], dtype=np.float32)

    fsw_rad = np.array([np.pi / 2.0, np.pi], dtype=np.float64)
    ds = xr.Dataset(
        coords={"camera_time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")},
        data_vars={"azimuth_angle": (("camera_time",), fsw_rad.astype(np.float32))},
    )
    config = GeolocationKernelConfig(dynamic_kernel_sources=["/tmp/test.bc"])

    result = apply_spice_azimuth_to_dataset(ds, config)

    mock_log_diff.assert_called_once()
    logged_fsw = mock_log_diff.call_args.args[1]
    np.testing.assert_allclose(logged_fsw, fsw_rad)
    np.testing.assert_array_equal(result["azimuth_angle"].values, np.array([90.0, 180.0], dtype=np.float32))


@patch("libera_cam.geolocation.calculate_azimuth_for_timestamps")
@patch("libera_cam.geolocation.KernelManager")
@patch("libera_cam.geolocation.prefetch_kernels")
def test_apply_spice_azimuth_to_dataset(mock_prefetch, mock_km_cls, mock_calc_az):
    mock_km = MagicMock()
    mock_km_cls.return_value = mock_km
    mock_km.__enter__.return_value = mock_km
    mock_km.__exit__.return_value = False
    mock_calc_az.return_value = np.array([10.0, 20.0], dtype=np.float32)

    ds = xr.Dataset(
        coords={"camera_time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")},
        data_vars={
            "WFOV_FSW_AZIMUTH_ANGLE": (("camera_time",), np.array([0.1, 0.2], dtype=np.float32)),
        },
    )
    config = GeolocationKernelConfig(dynamic_kernel_sources=["/tmp/test.bc"])

    result = apply_spice_azimuth_to_dataset(ds, config)

    mock_prefetch.assert_called_once_with(config)
    mock_calc_az.assert_called_once()
    np.testing.assert_array_equal(result["azimuth_angle"].values, np.array([10.0, 20.0], dtype=np.float32))
