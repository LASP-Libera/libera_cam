"""JPEG-LS decompression and 12/13-bit pixel split for WFOV L1A image blobs."""

from __future__ import annotations

import logging

import jpeg_ls
import numpy as np

logger = logging.getLogger(__name__)


def decompress_image(jpeg_ls_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decompress a JPEG-LS payload and separate DN from the integration mask.

    Parameters
    ----------
    jpeg_ls_bytes :
        Compressed JPEG-LS payload as stored in L1A ``WFOV_COMPRESSED_IMAGE``
        (headers already stripped; use ``WFOV_COMPRESSED_IMAGE_LENGTH`` for the valid length).

    Returns
    -------
    image_data : ndarray
        12-bit pixel values (int32), shape (PIXEL_COUNT_Y, PIXEL_COUNT_X).
    integration_mask : ndarray
        1-bit exposure mask from bit 12 (uint8), same shape as ``image_data``.
    """
    # jlsread returns a (rows, columns) array of the raw 13-bit sample values.
    try:
        raw_image_data = jpeg_ls.jlsread(jpeg_ls_bytes).astype(np.int32)
    except Exception as e:
        logger.error("JPEG-LS Decompression failed: %s", e)
        raise

    image_12bit = raw_image_data & 0x0FFF
    integration_mask = (raw_image_data >> 12) & 0x0001
    return image_12bit, integration_mask.astype("uint8")
