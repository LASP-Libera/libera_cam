"""Unit tests for L1A image parsing logic."""

from unittest.mock import patch

import numpy as np
import pytest

from libera_cam.image_parsing import l1a_parser
from libera_cam.image_parsing.exposure import actual_exposure_counts_to_ms, delta_exposure_counts_to_ms
from libera_cam.image_parsing.read_l1a_cam_data import _validate_execution_config, read_l1a_cam_data


def test_process_image_bits_logic():
    """Verify 12-bit vs 13th-bit extraction logic manually."""
    raw_data = np.array([0x1FFF, 0x0ABC], dtype=np.int32)

    image_12bit = raw_data & 0x0FFF
    integration_mask = (raw_data >> 12) & 0x0001

    expected_image = np.array([0xFFF, 0xABC], dtype=np.int32)
    expected_mask = np.array([1, 0], dtype=np.int32)

    np.testing.assert_array_equal(image_12bit, expected_image)
    np.testing.assert_array_equal(integration_mask, expected_mask)


@patch("libera_cam.image_parsing.l1a_parser.Image.open")
def test_decompress_image(mock_img_open):
    """Verify decompress_image splits bits from JPEG-LS payload bytes."""
    fake_raw_data = np.array([[0x1FFF, 0x0ABC]], dtype=np.int32)

    class MockImage:
        def __array__(self, dtype=None):
            return fake_raw_data.astype(dtype if dtype else np.int32)

        def close(self):
            pass

    mock_img_ctx = mock_img_open.return_value
    mock_img_ctx.__enter__.return_value = MockImage()

    img_data, mask_data = l1a_parser.decompress_image(b"fake_jpls_bytes")

    assert img_data[0, 0] == 0xFFF
    assert mask_data[0, 0] == 1
    assert img_data[0, 1] == 0xABC
    assert mask_data[0, 1] == 0


def test_actual_exposure_counts_to_ms():
    """FPGA actual exposure conversion matches L1A-documented equation."""
    value = 100.0
    expected = (value + 0.43 * 20) * 129.0 * 0.15625 / 1000.0
    assert actual_exposure_counts_to_ms(value) == pytest.approx(expected)


def test_delta_exposure_counts_to_ms():
    """DELTA_EXP conversion uses clock period only."""
    value = 8000.0
    expected = value * 0.15625 / 1000.0
    assert delta_exposure_counts_to_ms(value) == pytest.approx(expected)


def test_read_l1a_cam_data_rejects_packet_only():
    """Packet-only L1A (pre-5.10.5) must raise a clear error."""
    import xarray as xr

    ds = xr.Dataset({"ICIE__WFOV_DATA": (("PACKET",), np.zeros(4, dtype="|S972"))})
    with pytest.raises(ValueError, match="image-centric"):
        read_l1a_cam_data(ds)


def test_validate_execution_config():
    """Chunk size below 1 is rejected."""
    with pytest.raises(ValueError, match="LIBERA_CAM_CHUNK_SIZE must be >= 1"):
        _validate_execution_config(0)

    with pytest.raises(ValueError, match="LIBERA_CAM_CHUNK_SIZE must be >= 1"):
        _validate_execution_config(-5)

    _validate_execution_config(10)
