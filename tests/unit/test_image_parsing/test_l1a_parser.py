"""Unit tests for L1A image parsing logic."""

from unittest.mock import patch

import numpy as np

from libera_cam.image_parsing import l1a_parser


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
