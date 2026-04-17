"""Unit tests for L1A image parsing logic."""

from unittest.mock import MagicMock, patch

import numpy as np

from libera_cam.image_parsing import l1a_parser


def test_process_image_bits_logic():
    """Verify 12-bit vs 13th-bit extraction logic manually."""
    # Create a synthetic image array
    # Value 0x1FFF = 0001 1111 1111 1111 (13 bits set)
    # Lower 12 bits = 0xFFF = 4095
    # 13th bit = 1

    # Value 0x0ABC (No 13th bit)
    # Lower 12 bits = 0xABC
    # 13th bit = 0

    raw_data = np.array([0x1FFF, 0x0ABC], dtype=np.int32)

    # Mimic the logic in decompress_image
    image_12bit = raw_data & 0x0FFF
    integration_mask = (raw_data >> 12) & 0x0001

    expected_image = np.array([0xFFF, 0xABC], dtype=np.int32)
    expected_mask = np.array([1, 0], dtype=np.int32)

    np.testing.assert_array_equal(image_12bit, expected_image)
    np.testing.assert_array_equal(integration_mask, expected_mask)


@patch("libera_cam.image_parsing.l1a_parser.extract_dict_from_bytearray")
def test_parse_image_metadata_structure(mock_extract):
    """Verify parse_image_metadata removes heavy payloads."""
    # Setup mock return value
    mock_extract.return_value = {"some_meta": 123, "compressed_image_data": b"heavy_data", "raw_footer": b"footer_data"}

    blob = bytearray(b"dummy_blob")
    result = l1a_parser.parse_image_metadata(blob)

    # Assert calls
    mock_extract.assert_called_once_with(blob)

    # Assert heavy keys are removed
    assert "compressed_image_data" not in result
    assert "raw_footer" not in result
    # Assert other keys remain
    assert result["some_meta"] == 123


@patch("libera_cam.image_parsing.l1a_parser.Image.open")
@patch("libera_cam.image_parsing.l1a_parser.extract_dict_from_bytearray")
def test_decompress_image(mock_extract, mock_img_open):
    """Verify decompress_image calls proper helpers and splits bits."""
    mock_extract.return_value = {"compressed_image_data": b"fake_jpls_bytes"}

    # We need to mock the context manager behavior of Image.open
    mock_img = MagicMock()
    # When np.array(img) is called, it iterates or uses __array__ interface.
    fake_raw_data = np.array([[0x1FFF, 0x0ABC]], dtype=np.int32)

    # Setup the context manager to return our mock image
    mock_img_ctx = mock_img_open.return_value
    mock_img_ctx.__enter__.return_value = mock_img

    mock_img.__array__ = lambda *args, **kwargs: fake_raw_data

    class MockImage:
        def __array__(self, dtype=None):
            return fake_raw_data.astype(dtype if dtype else np.int32)

        def close(self):
            pass

    mock_img_ctx.__enter__.return_value = MockImage()

    # Act
    blob = bytearray(b"dummy")
    img_data, mask_data = l1a_parser.decompress_image(blob)

    # Assert
    mock_extract.assert_called_once_with(blob)

    # Check values
    # 0x1FFF -> Data 0xFFF, Mask 1
    # 0x0ABC -> Data 0xABC, Mask 0
    assert img_data[0, 0] == 0xFFF
    assert mask_data[0, 0] == 1
    assert img_data[0, 1] == 0xABC
    assert mask_data[0, 1] == 0
