"""Unit tests for L1A image parsing logic."""

from unittest.mock import patch

import numpy as np

from libera_cam.image_parsing import l1a_parser


@patch("libera_cam.image_parsing.l1a_parser.jpeg_ls.jlsread")
def test_decompress_image(mock_jlsread):
    """Verify decompress_image splits bits from JPEG-LS payload bytes."""
    # jlsread returns the raw 13-bit samples as an unsigned integer array
    mock_jlsread.return_value = np.array([[0x1FFF, 0x0ABC]], dtype=np.uint16)

    img_data, mask_data = l1a_parser.decompress_image(b"fake_jpls_bytes")

    mock_jlsread.assert_called_once_with(b"fake_jpls_bytes")
    assert img_data[0, 0] == 0xFFF
    assert mask_data[0, 0] == 1
    assert img_data[0, 1] == 0xABC
    assert mask_data[0, 1] == 0
