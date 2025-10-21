"""
Image Metadata Parser
Extracts FSW and FPGA metadata from binary image files or bytearrays.
"""

import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ParserConfig:
    """Configuration for metadata parser."""

    fpga_header_size: int = 140
    fsw_header_size: int = 36
    validate_crc: bool = False
    log_level: str = "INFO"
    skip_errors_default: bool = True

    @property
    def min_file_size(self) -> int:
        """Minimum required file size."""
        return self.fsw_header_size + self.fpga_header_size

    def __post_init__(self):
        """Set logging level after initialization."""
        logging.getLogger(__name__).setLevel(self.log_level)


# Default configuration instance
DEFAULT_CONFIG = ParserConfig()


# ============================================================================
# Constants and Enums
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
    """
    Calculate CRC32 checksum for data.

    Args:
        data: Bytes to calculate CRC for

    Returns:
        CRC32 checksum as unsigned 32-bit integer

    Note:
        Uses standard CRC32 algorithm. Modify if hardware uses different polynomial.
    """
    import zlib

    # Standard CRC32 - returns signed int, convert to unsigned
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return crc


def verify_crc(metadata: dict, image_data: bytes, config: ParserConfig = DEFAULT_CONFIG) -> bool:
    """
    Verify CRC matches calculated value for image data.

    Args:
        metadata: Parsed metadata containing CRC field
        image_data: Raw image data to validate
        config: Parser configuration

    Returns:
        True if CRC matches, False otherwise

    Raises:
        CRCValidationError: If config.validate_crc is True and CRC doesn't match
    """
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
    """
    Swap 32-bit words in the data.

    Args:
        data: Input bytes to swap

    Returns:
        Bytearray with swapped 32-bit words
    """
    if len(data) % 4 != 0:
        logger.warning(f"Data length {len(data)} is not divisible by 4")

    result = bytearray(len(data))
    for i in range(0, len(data), 4):
        result[i : i + 4] = data[i : i + 4][::-1]
    return result


def read_fsw_metadata(file: BinaryIO) -> dict:
    """
    Read FSW metadata (36 bytes).

    Args:
        file: File-like object positioned at start of FSW metadata

    Returns:
        Dictionary containing FSW metadata fields

    Raises:
        MetadataParseError: If unable to read expected bytes
    """
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
        metadata["obs_id"] = struct.unpack(">I", file.read(4))[0]
        metadata["commanded_exp_time_1"] = struct.unpack(">I", file.read(4))[0]
        metadata["commanded_exp_time_2"] = struct.unpack(">I", file.read(4))[0]
        metadata["azimuth_angle"] = struct.unpack(">f", file.read(4))[0]

        return metadata

    except struct.error as e:
        raise MetadataParseError(f"Failed to parse FSW metadata: {e}")


def read_fpga_metadata(data: bytes, config: ParserConfig = DEFAULT_CONFIG) -> tuple[dict, dict, dict]:
    """
    Read FPGA header, footer, and status metadata (140 bytes).

    Args:
        data: FPGA metadata bytes
        config: Parser configuration

    Returns:
        Tuple of (header_metadata, footer_metadata, status_metadata)

    Raises:
        MetadataParseError: If data is invalid
    """
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

        # Footer metadata
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
        status_meta["spare"] = (fpga_status >> 7) & 0x01

        return header_meta, footer_meta, status_meta

    except (ValueError, IndexError) as e:
        raise MetadataParseError(f"Failed to parse FPGA metadata: {e}")


def extract_metadata(
    source: str | Path | bytes | bytearray, config: ParserConfig = DEFAULT_CONFIG, image_data: bytes | None = None
) -> dict:
    """
    Extract all metadata from a file path or bytearray.

    Args:
        source: File path (str/Path) or binary data (bytes/bytearray)
        config: Parser configuration
        image_data: Optional image data for CRC validation

    Returns:
        Dictionary containing all metadata fields

    Raises:
        MetadataParseError: If parsing fails
        FileNotFoundError: If file path doesn't exist
        ValueError: If source type is invalid
        CRCValidationError: If CRC validation fails and config.validate_crc is True
    """
    # Handle different input types
    if isinstance(source, (str, Path)):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        file_size = source.stat().st_size
        if file_size < config.min_file_size:
            raise MetadataParseError(f"File too small: {file_size} bytes (minimum {config.min_file_size})")

        with open(source, "rb") as f:
            fsw_meta = read_fsw_metadata(f)
            fpga_data = f.read(config.fpga_header_size)
            # If no image_data provided and we want CRC validation, read it
            if image_data is None and config.validate_crc:
                image_data = f.read()

    elif isinstance(source, (bytes, bytearray)):
        if len(source) < config.min_file_size:
            raise MetadataParseError(f"Data too small: {len(source)} bytes (minimum {config.min_file_size})")

        bio = BytesIO(source)
        fsw_meta = read_fsw_metadata(bio)
        fpga_data = bio.read(config.fpga_header_size)
        # If no image_data provided and we want CRC validation, use remaining data
        if image_data is None and config.validate_crc:
            image_data = bio.read()

    else:
        raise ValueError(f"Invalid source type: {type(source)}. Expected str, Path, bytes, or bytearray")

    # Parse FPGA metadata
    header_meta, footer_meta, status_meta = read_fpga_metadata(fpga_data, config)

    # Combine all metadata
    combined = {}
    combined.update(fsw_meta)
    combined.update(header_meta)
    combined.update(footer_meta)
    combined.update(status_meta)

    # Add decoded lookup values using enums
    combined["bitmask_id_name"] = BitmaskID.get_name(combined["bitmask_id"])
    combined["img_mode_name"] = ImageMode.get_name(combined["img_mode"])
    combined["readout_name"] = ReadoutMode.get_name(combined["readout"])

    # Validate CRC if image data provided
    if image_data is not None:
        combined["crc_valid"] = verify_crc(combined, image_data, config)

    return combined


def validate_metadata(metadata: dict) -> list[str]:
    """
    Validate metadata for common issues.

    Args:
        metadata: Metadata dictionary to validate

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Check for error flags
    error_flags = ["sync_error", "pid_error", "size_error", "eop_error", "eep_error", "crc_error", "drop_error"]
    active_errors = [flag for flag in error_flags if metadata.get(flag, 0) == 1]
    if active_errors:
        warnings.append(f"Error flags set: {', '.join(active_errors)}")

    # Check CRC validation result
    if "crc_valid" in metadata and not metadata["crc_valid"]:
        warnings.append("CRC validation failed")

    # Check image count vs total
    if metadata.get("image_count", 0) > metadata.get("image_total", 0):
        warnings.append(f"Image count ({metadata['image_count']}) exceeds total ({metadata['image_total']})")

    # Check for unusual values
    if metadata.get("width", 0) == 0 or metadata.get("height", 0) == 0:
        warnings.append("Image dimensions are zero")

    if metadata.get("image_length", 0) == 0:
        warnings.append("Image length is zero")

    return warnings


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Using custom configuration
    custom_config = ParserConfig(validate_crc=True, log_level="DEBUG", skip_errors_default=False)

    # Example 2: Single file with CRC validation
    try:
        metadata = extract_metadata("image_001.dat", config=custom_config)
        warnings = validate_metadata(metadata)
        if warnings:
            print("Warnings:", warnings)
        df = pd.DataFrame(metadata)
        print(df.T)
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Single bytearray with image data for CRC check
    # with open('image_001.dat', 'rb') as f:
    #     full_data = f.read()
    # metadata_size = DEFAULT_CONFIG.min_file_size
    # metadata = extract_metadata(
    #     full_data[:metadata_size],
    #     image_data=full_data[metadata_size:]
    # )

    # Example 4: Multiple files with validation disabled
    # no_crc_config = ParserConfig(validate_crc=False)
    # filepaths = ['image_001.dat', 'image_002.dat', 'image_003.dat']
    # df = process_multiple_sources(filepaths, config=no_crc_config)
    # print(df[['obs_id', 'image_count', 'crc_valid']])

    # Example 5: Access enum values
    # print(f"Available bitmask IDs: {[e.name for e in BitmaskID]}")
    # print(f"Bitmask ID 8 is: {BitmaskID.get_name(8)}")
