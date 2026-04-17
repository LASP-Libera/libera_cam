import io
import logging

import numpy as np
import pillow_jpls  # noqa: F401 - Register JPEG-LS plugin
from PIL import Image

from libera_cam.image_parsing.metadata_parser import extract_dict_from_bytearray

logger = logging.getLogger(__name__)


def parse_image_metadata(blob: bytearray) -> dict:
    """
    Extracts metadata from the binary blob.

    Args:
        blob (bytearray): The reassembled binary packet.

    Returns:
        dict: The extracted metadata dictionary.
              Note: 'compressed_image_data' and 'raw_footer' are removed to save memory.
    """
    meta = extract_dict_from_bytearray(blob)
    # Remove heavy payload keys to conserve memory
    if "compressed_image_data" in meta:
        del meta["compressed_image_data"]
    if "raw_footer" in meta:
        del meta["raw_footer"]
    return meta


def decompress_image(blob: bytearray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompresses the JPEG-LS image from the blob and separates pixel data from integration mask.

    Args:
        blob (bytearray): The reassembled binary packet.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - image_data (np.ndarray): 12-bit pixel values (int32).
            - integration_mask (np.ndarray): 1-bit mask (uint8).
    """
    # Extract payload from the blob
    # We rely on extract_dict_from_bytearray to parse headers and isolate the payload safely.
    full_data = extract_dict_from_bytearray(blob)
    compressed_bytes = full_data["compressed_image_data"]

    # Decompress Image
    with io.BytesIO(compressed_bytes) as bytes_io:
        try:
            with Image.open(bytes_io) as img:
                raw_image_data = np.array(img, dtype=np.int32)
        except Exception as e:
            logger.error(f"JPEG-LS Decompression failed: {e}")
            raise

    # Process 12-bit Data vs 13th-bit Mask
    # Mask 0x0FFF gets the lower 12 bits (Pixel value)
    image_12bit = raw_image_data & 0x0FFF

    # Shift right 12 bits and mask 1 bit to get the 13th bit (Integration Mask)
    integration_mask = (raw_image_data >> 12) & 0x0001

    return image_12bit, integration_mask.astype("uint8")
