"""
Image Metadata Parser
Extracts FSW, FPGA metadata, and Compressed Payload from binary image blobs.
"""

import logging
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import BinaryIO

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Constants
# ============================================================================


@dataclass
class ParserConfig:
    """Configuration for metadata parser."""

    fpga_header_size: int = 140
    fsw_header_size: int = 36
    fpga_footer_size: int = 8
    validate_crc: bool = False
    log_level: str = "INFO"
    skip_errors_default: bool = True

    @property
    def min_file_size(self) -> int:
        """Minimum required file size."""
        return self.fsw_header_size + self.fpga_header_size + self.fpga_footer_size

    def __post_init__(self):
        """Set logging level after initialization."""
        logging.getLogger(__name__).setLevel(self.log_level)


# Default configuration instance
DEFAULT_CONFIG = ParserConfig()


# ============================================================================
# Enums
# ============================================================================


class BitmaskID(IntEnum):
    """Bitmask partition identifiers."""

    FF = 0
    NADIR_STRIPE = 1
    FORWARD_STRIPE = 2
    CROSS_STRIPE = 4
    ALL_STRIPES = 7
    ADM = 8
    STRIPES_ADM = 15
    LIMB = 16
    CUSTOM_0 = 32
    CUSTOM_1 = 33
    CUSTOM_2 = 34
    CUSTOM_3 = 35
    LOCK_ON = 64
    RAPS_STRIPE = 128
    RAPS_ADM = 137

    @classmethod
    def get_name(cls, value: int) -> str:
        """Get name for bitmask ID, return UNKNOWN if not found."""
        try:
            return cls(value).name
        except ValueError:
            return "UNKNOWN"


class ImageMode(IntEnum):
    """Imaging mode identifiers."""

    DUAL = 0
    VIDEO = 1
    IMGA = 2
    IMGB = 3

    @classmethod
    def get_name(cls, value: int) -> str:
        """Get name for image mode, return UNKNOWN if not found."""
        try:
            return cls(value).name
        except ValueError:
            return "UNKNOWN"


class ReadoutMode(IntEnum):
    """Readout mode identifiers."""

    SINGLE = 0
    DUAL = 1

    @classmethod
    def get_name(cls, value: int) -> str:
        """Get name for readout mode, return UNKNOWN if not found."""
        try:
            return cls(value).name
        except ValueError:
            return "UNKNOWN"


# ============================================================================
# Exceptions
# ============================================================================


class MetadataParseError(Exception):
    """Custom exception for metadata parsing errors."""

    pass


class CRCValidationError(Exception):
    """Custom exception for CRC validation failures."""

    pass


# ============================================================================
# CRC Calculation
# ============================================================================


def calculate_crc32(data: bytes) -> int:
    """Calculate CRC32 checksum for data."""
    # Standard CRC32 - returns signed int, convert to unsigned
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return crc


def verify_crc(metadata: dict, image_data: bytes, config: ParserConfig = DEFAULT_CONFIG) -> bool:
    """Verify CRC matches calculated value for image data."""
    if "crc" not in metadata:
        logger.warning("No CRC field in metadata")
        return False

    expected_crc = metadata["crc"]
    calculated_crc = calculate_crc32(image_data)

    match = expected_crc == calculated_crc

    if not match:
        error_msg = f"CRC mismatch: expected 0x{expected_crc:08X}, calculated 0x{calculated_crc:08X}"
        if config.validate_crc:
            raise CRCValidationError(error_msg)
        else:
            logger.warning(error_msg)

    return match


# ============================================================================
# Parsing Functions
# ============================================================================


def swap_32bit_words(data: bytes) -> bytearray:
    """Swap 32-bit words in the data."""
    if len(data) % 4 != 0:
        logger.warning(f"Data length {len(data)} is not divisible by 4")

    result = bytearray(len(data))
    for i in range(0, len(data), 4):
        result[i : i + 4] = data[i : i + 4][::-1]
    return result


def read_fsw_metadata(file: BinaryIO) -> dict:
    """Read FSW metadata (36 bytes)."""
    try:
        metadata = {}
        metadata["fsw_length"] = struct.unpack("B", file.read(1))[0]

        # Bit-packed byte
        second_byte = struct.unpack("B", file.read(1))[0]
        metadata["jpeg_bypass"] = (second_byte >> 7) & 1
        metadata["bitmask_disable"] = (second_byte >> 6) & 1
        metadata["testpattern"] = (second_byte >> 5) & 1
        metadata["bitmask_id"] = (second_byte >> 3) & 0x03
        metadata["img_mode"] = (second_byte >> 1) & 0x03

        metadata["pixel_mask_id"] = struct.unpack("B", file.read(1))[0]
        metadata["simulator"] = struct.unpack("B", file.read(1))[0]
        metadata["cadence"] = struct.unpack(">H", file.read(2))[0]
        metadata["image_total"] = struct.unpack("B", file.read(1))[0]
        metadata["image_count"] = struct.unpack("B", file.read(1))[0]
        metadata["flash_write_pointer"] = struct.unpack(">I", file.read(4))[0]
        metadata["timestamp_seconds"] = struct.unpack(">I", file.read(4))[0]
        metadata["timestamp_subseconds"] = struct.unpack(">I", file.read(4))[0]
        metadata["rad_obs_id"] = struct.unpack(">H", file.read(2))[0]
        metadata["cam_obs_id"] = struct.unpack(">H", file.read(2))[0]
        metadata["commanded_exp_time_1"] = struct.unpack(">I", file.read(4))[0]
        metadata["commanded_exp_time_2"] = struct.unpack(">I", file.read(4))[0]
        metadata["azimuth_angle"] = struct.unpack(">f", file.read(4))[0]

        return metadata

    except struct.error as e:
        raise MetadataParseError(f"Failed to parse FSW metadata: {e}")


def read_fpga_metadata(data: bytes, config: ParserConfig = DEFAULT_CONFIG) -> tuple[dict, dict, dict]:
    """Read FPGA header, footer, and status metadata (140 bytes)."""
    if len(data) != config.fpga_header_size:
        raise MetadataParseError(f"Expected {config.fpga_header_size} bytes, got {len(data)}")

    try:
        data = swap_32bit_words(data)
        header = data[2:100][::2]  # Take every other byte
        footer = data[100:136][::2]

        # Header metadata
        header_meta = {}
        header_meta["image_length"] = int.from_bytes(header[0:4], byteorder="little")
        header_meta["flags"] = int.from_bytes(header[4:5], byteorder="little")
        header_meta["frame_id"] = int.from_bytes(header[5:6], byteorder="little")
        header_meta["tag"] = int.from_bytes(header[6:14], byteorder="little")
        header_meta["actual_exp_time_1"] = int.from_bytes(header[14:17], byteorder="little")
        header_meta["temperature"] = int.from_bytes(header[17:19], byteorder="little")
        header_meta["gain"] = int.from_bytes(header[19:20], byteorder="little")
        header_meta["width"] = int.from_bytes(header[20:22], byteorder="little")
        header_meta["height"] = int.from_bytes(header[22:24], byteorder="little")
        header_meta["offset_x"] = int.from_bytes(header[24:26], byteorder="little")
        header_meta["offset_y"] = int.from_bytes(header[26:28], byteorder="little")
        header_meta["readout"] = int.from_bytes(header[28:29], byteorder="little")
        header_meta["actual_exp_time_2"] = int.from_bytes(header[29:32], byteorder="little")
        header_meta["delta"] = int.from_bytes(header[32:35], byteorder="little")
        header_meta["exposure_step"] = int.from_bytes(header[35:38], byteorder="little")
        header_meta["nr_slopes"] = int.from_bytes(header[38:39], byteorder="little")
        header_meta["kp1"] = int.from_bytes(header[39:42], byteorder="little")
        header_meta["kp2"] = int.from_bytes(header[42:45], byteorder="little")
        header_meta["vlow_3"] = int.from_bytes(header[45:46], byteorder="little")
        header_meta["vlow_2"] = int.from_bytes(header[46:47], byteorder="little")
        header_meta["exp_seq"] = int.from_bytes(header[47:48], byteorder="little")
        header_meta["footer_size"] = int.from_bytes(header[48:49], byteorder="little")

        # Footer metadata (Internal to FPGA block)
        footer_meta = {}
        footer_meta["pixel_sum"] = int.from_bytes(footer[0:4], byteorder="little")
        footer_meta["dark"] = int.from_bytes(footer[4:7], byteorder="little")
        footer_meta["white"] = int.from_bytes(footer[7:10], byteorder="little")
        footer_meta["footer_delta"] = int.from_bytes(footer[10:14], byteorder="little")
        footer_meta["crc"] = int.from_bytes(footer[14:18], byteorder="little")

        # FPGA status (bit-packed error flags)
        fpga_status = int.from_bytes(data[136:140], byteorder="little")
        status_meta = {}
        status_meta["sync_error"] = (fpga_status >> 0) & 0x01
        status_meta["pid_error"] = (fpga_status >> 1) & 0x01
        status_meta["size_error"] = (fpga_status >> 2) & 0x01
        status_meta["eop_error"] = (fpga_status >> 3) & 0x01
        status_meta["eep_error"] = (fpga_status >> 4) & 0x01
        status_meta["crc_error"] = (fpga_status >> 5) & 0x01
        status_meta["drop_error"] = (fpga_status >> 6) & 0x01

        return header_meta, footer_meta, status_meta

    except (ValueError, IndexError) as e:
        raise MetadataParseError("Failed to parse FPGA metadata") from e


def extract_dict_from_bytearray(source: bytearray, config: ParserConfig = DEFAULT_CONFIG) -> dict:
    """
    Extract all metadata and the compressed image payload from a source.

    The source is expected to follow this structure:
    [FSW Header (36B)] [FPGA Header (140B)] [Compressed Payload (Variable)] [FPGA Footer (8B)]

    Returns:
        dict: A dictionary containing:
            - All metadata keys from FSW and FPGA headers.
            - "compressed_image_data": The isolated compressed image bytes.
            - "raw_footer": The last 8 bytes of the file.
            - "crc_valid": Boolean result of CRC check (if enabled).
    """

    # Size Validation
    if len(source) < config.min_file_size:
        raise MetadataParseError(f"Data too small: {len(source)} bytes (minimum {config.min_file_size})")

    # Component Slicing
    fsw_end = config.fsw_header_size
    fpga_header_end = fsw_end + config.fpga_header_size
    footer_start = len(source) - config.fpga_footer_size

    # Sanity check for negative payload size
    if footer_start < fpga_header_end:
        raise MetadataParseError("File structure invalid: Overlapping headers and footers (Negative payload size).")

    fsw_bytes = source[0:fsw_end]
    fpga_header_bytes = source[fsw_end:fpga_header_end]
    compressed_image_bytes = source[fpga_header_end:footer_start]
    footer_bytes = source[footer_start:]

    # Parse Components
    with BytesIO(fsw_bytes) as bio:
        fsw_meta = read_fsw_metadata(bio)

    header_meta, footer_meta, status_meta = read_fpga_metadata(fpga_header_bytes, config)

    # Combine Results
    combined = {}
    combined.update(fsw_meta)
    combined.update(header_meta)
    combined.update(footer_meta)
    combined.update(status_meta)

    # Add decodes
    combined["bitmask_id_name"] = BitmaskID.get_name(combined["bitmask_id"])
    combined["img_mode_name"] = ImageMode.get_name(combined["img_mode"])
    combined["readout_name"] = ReadoutMode.get_name(combined["readout"])

    # Add Raw Data Components
    combined["compressed_image_data"] = compressed_image_bytes
    combined["raw_footer"] = footer_bytes

    # CRC Validation
    # We use the extracted payload for validation
    if config.validate_crc:
        combined["crc_valid"] = verify_crc(combined, compressed_image_bytes, config)
    else:
        combined["crc_valid"] = False

    return combined
