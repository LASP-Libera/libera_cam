import io
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL.Image as Image
from space_packet_parser.xarr import create_dataset
from space_packet_parser.xtce.definitions import XtcePacketDefinition

from libera_cam.utils.metadata_parser import extract_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TimestampConfig:
    """Configuration for timestamp field extraction."""

    packet_day: str
    packet_ms: str
    packet_us: str
    start_hi: str | None = None
    start_lo: str | None = None


@dataclass
class SampleConfig:
    """Configuration for sample-based data extraction."""

    sample_count: int
    field_patterns: dict[str, str]  # Maps output column name to field pattern template


@dataclass
class ArrayFieldConfig:
    """Configuration for array-like fields (hybrid approach)."""

    field_name: str  # Output column name (e.g., 'science_data')
    pattern: str  # Field pattern template (e.g., 'ICIE__WFOV_DATA_{index}')
    array_size: int  # Number of elements in the array
    data_type: str  # Expected data type for documentation


@dataclass
class APIDConfig:
    """Complete configuration for an APID."""

    apid: int
    name: str
    packet_name: str
    has_samples: bool
    timestamp_config: TimestampConfig
    sample_config: SampleConfig | None = None
    array_fields: list[ArrayFieldConfig] | None = None  # For hybrid array processing
    additional_fields: dict[str, str] | None = None  # Static fields to extract


APID_CONFIGS = {
    1040: APIDConfig(
        apid=1040,
        name="icie_wfov_sci",
        packet_name="wfov_sci",
        has_samples=False,  # Using hybrid array approach instead
        timestamp_config=TimestampConfig(
            packet_day="ICIE__TM_DAY_WFOV_SCI", packet_ms="ICIE__TM_MS_WFOV_SCI", packet_us="ICIE__TM_US_WFOV_SCI"
        ),
        array_fields=[
            ArrayFieldConfig(
                field_name="mem_dump_names", pattern="ICIE__MEM_DUMP_NAME_WFOV_{index}", array_size=8, data_type="C8"
            ),
            ArrayFieldConfig(
                field_name="science_data", pattern="ICIE__WFOV_DATA_{index}", array_size=972, data_type="U8"
            ),
        ],
        additional_fields={
            "mem_dump_offset": "ICIE__MEM_DUMP_OFFSET_WFOV",
            "mem_dump_length": "ICIE__MEM_DUMP_LENGTH_WFOV",
            "mem_dump_trans": "ICIE__MEM_DUMP_TRANS_WFOV",
            "mem_dump_flags": "ICIE__MEM_DUMP_FLAGS_WFOV",
            "SOURCE_FILE": None,
        },
    )
}

APID_NAMES = {1040: "icie_wfov_sci"}

# Standard CCSDS header fields present in all packets
CCSDS_HEADER_FIELDS = ["VERSION", "TYPE", "SEC_HDR_FLAG", "PKT_APID", "SEQ_FLGS", "SRC_SEQ_CTR", "PKT_LEN"]


def get_apid_config(apid: int) -> APIDConfig:
    """Get configuration for a specific APID.

    Args:
        apid: The APID number

    Returns:
        APIDAConfig object for the specified APID

    Raises:
        ValueError: If APID is not supported
    """
    if apid not in APID_CONFIGS:
        supported_apids = list(APID_CONFIGS.keys())
        raise ValueError(f"APID {apid} not supported. Supported APIDs: {supported_apids}")

    return APID_CONFIGS[apid]


def tai_epoch_to_datetime(seconds, microseconds=0, utc_offset_hours=0):
    """
    Convert TAI seconds since Jan 1, 1958 to datetime.

    Args:
        seconds: Seconds since TAI epoch (Jan 1, 1958)
        microseconds: Additional microseconds
        utc_offset_hours: UTC offset in hours (0 for UTC, -6 for MDT, etc.)

    Returns:
        datetime object in specified timezone
    """
    from datetime import timezone

    tai_epoch = datetime(1958, 1, 1, 0, 0, 0)
    total_seconds = seconds + microseconds / 1_000_000
    utc_time = tai_epoch + timedelta(seconds=total_seconds)

    tz = timezone(timedelta(hours=utc_offset_hours))
    return utc_time.replace(tzinfo=UTC).astimezone(tz)


def tai_epoch_from_day_ms_us(day_offset, milliseconds, microseconds, utc_offset_hours=0):
    """
    Convert day offset + milliseconds + microseconds to datetime.

    Args:
        day_offset: Days since Jan 1, 1958 (TAI epoch)
        milliseconds: Milliseconds within the day
        microseconds: Additional microseconds
        utc_offset_hours: UTC offset in hours (0 for UTC, -6 for MDT, etc.)

    Returns:
        datetime object in specified timezone
    """
    # Convert day offset to seconds and add milliseconds/microseconds
    total_seconds = day_offset * 86400 + milliseconds / 1000.0
    total_microseconds = microseconds

    return tai_epoch_to_datetime(total_seconds, total_microseconds, utc_offset_hours)


def colorado_datetime_to_utc_filename_format(dt_colorado):
    """
    Convert Colorado datetime to UTC filename format (YYYY_DDD_HH_MM_SS).

    Args:
        dt_colorado: datetime object assumed to be in Colorado time (unless timezone-aware)

    Returns:
        string in format "YYYY_DDD_HH_MM_SS" where DDD is day of year, time converted to UTC
    """
    if dt_colorado is None:
        return None

    from datetime import timedelta, timezone

    if dt_colorado.tzinfo is None:
        # Assume naive datetime is in Colorado time
        # Colorado is typically UTC-7 (MST) or UTC-6 (MDT)
        # For simplicity, assume MDT (UTC-6) - adjust if needed
        colorado_tz = timezone(timedelta(hours=-6))
        colorado_aware = dt_colorado.replace(tzinfo=colorado_tz)
        utc_dt = colorado_aware.astimezone(UTC)
    else:
        # If timezone-aware, convert whatever timezone it is to UTC
        utc_dt = dt_colorado.astimezone(UTC)

    year = utc_dt.year
    day_of_year = utc_dt.timetuple().tm_yday
    hour = utc_dt.hour
    minute = utc_dt.minute
    second = utc_dt.second

    return f"{year:04d}_{day_of_year:03d}_{hour:02d}_{minute:02d}_{second:02d}"


def extract_ccsds_header(packet_row: pd.Series) -> dict[str, Any]:
    """Extract CCSDS header fields from a packet row."""
    header = {}
    for field in CCSDS_HEADER_FIELDS:
        header[field] = packet_row.get(field, "N/A")
    return header


def extract_timestamp_data(packet_row: pd.Series, config: APIDConfig, source_file: str) -> dict[str, Any]:
    """Extract timestamp and metadata from a packet row."""
    ts_config = config.timestamp_config

    metadata = {
        "PKT_DAY": packet_row.get(ts_config.packet_day, "N/A"),
        "PKT_MS": packet_row.get(ts_config.packet_ms, "N/A"),
        "PKT_US": packet_row.get(ts_config.packet_us, "N/A"),
        "SOURCE_FILE": source_file,
    }

    # Add start time fields if they exist
    if ts_config.start_hi:
        metadata["PKT_SAMP_START_HI"] = packet_row.get(ts_config.start_hi, "N/A")
    if ts_config.start_lo:
        metadata["PKT_SAMP_START_LO"] = packet_row.get(ts_config.start_lo, "N/A")

    # Add additional fields if configured
    if config.additional_fields:
        for field_name, field_key in config.additional_fields.items():
            if field_key:  # Skip None values (like SOURCE_FILE)
                metadata[field_name] = packet_row.get(field_key, "N/A")
            elif field_name == "SOURCE_FILE":
                metadata[field_name] = source_file

    return metadata


def extract_array_field(packet_row: pd.Series, array_config: ArrayFieldConfig) -> list[Any]:
    """
    Extract array data from packet row using array field configuration.

    Args:
        packet_row: Pandas Series containing packet data
        array_config: ArrayFieldConfig specifying how to extract the array

    Returns:
        List containing the extracted array data
    """
    array_data = []
    for i in range(array_config.array_size):
        field_name = array_config.pattern.format(index=i)
        value = packet_row.get(field_name, None)
        array_data.append(value)
    return array_data


def extract_array_fields(packet_row: pd.Series, config: APIDConfig) -> dict[str, list[Any]]:
    """
    Extract all array fields from a packet row.

    Args:
        packet_row: Pandas Series containing packet data
        config: APID configuration

    Returns:
        Dictionary mapping array field names to extracted arrays
    """
    array_data = {}

    if config.array_fields:
        for array_config in config.array_fields:
            array_data[array_config.field_name] = extract_array_field(packet_row, array_config)

    return array_data


def calculate_timestamp_for_packet(metadata: dict[str, Any], utc_offset_hours=0) -> datetime | None:
    """Calculate timestamp for a sample based on APID-specific logic."""
    # Fallback to packet-level day/ms/us (for housekeeping data like 1057, new APIDs)
    pkt_day = metadata.get("PKT_DAY")
    pkt_ms = metadata.get("PKT_MS")
    pkt_us = metadata.get("PKT_US")

    if (
        pkt_day is not None
        and pkt_ms is not None
        and pkt_us is not None
        and pkt_day != "N/A"
        and pkt_ms != "N/A"
        and pkt_us != "N/A"
    ):
        return tai_epoch_from_day_ms_us(pkt_day, pkt_ms, pkt_us, utc_offset_hours)


def process_non_sample_apid(
    raw_df: pd.DataFrame, config: APIDConfig, source_file: str, utc_offset_hours=0
) -> list[dict[str, Any]]:
    """Process packets without sample data (APIDs 1035, 1043, 1057, 1059, housekeeping, and hybrid array APIDs)."""
    records_list = []

    for _, packet_row in raw_df.iterrows():
        # Extract common packet data
        ccsds_header = extract_ccsds_header(packet_row)
        metadata = extract_timestamp_data(packet_row, config, source_file)

        # For non-sample data, we create one record per packet
        # Use the same timestamp calculation logic as samples
        timestamp = calculate_timestamp_for_packet(metadata, utc_offset_hours)

        # Start building the record
        record = {"TIMESTAMP": timestamp, **ccsds_header, **metadata}

        # Handle array fields for hybrid approach (e.g., APID 1040)
        if config.array_fields:
            array_data = extract_array_fields(packet_row, config)
            record.update(array_data)

        records_list.append(record)

    return records_list


def parse_apid_packets(packet_files: list[Path], xtce_file: Path, apid: int, utc_offset_hours=0) -> pd.DataFrame:
    """
    Parse packets for a specific APID using configuration-driven approach.

    Args:
        packet_files: List of packet files to parse
        xtce_file: Path to XTCE definition file
        apid: APID number to parse
        utc_offset_hours: UTC offset in hours (0 for UTC, -6 for MDT, etc.)

    Returns:
        DataFrame with parsed data for the specified APID
    """
    logger.info(f"Parsing APID {apid}")

    # Get configuration for this APID
    try:
        config = get_apid_config(apid)
    except ValueError as e:
        logger.error(str(e))
        return pd.DataFrame()

    # Load XTCE definition
    definition = XtcePacketDefinition.from_xtce(xtce_file)

    # Parse the packet files
    dataset_dict = create_dataset(
        packet_files=packet_files,
        xtce_packet_definition=definition,
        generator_kwargs={"skip_header_bytes": 8},  # Adjust if needed
    )

    logger.info(f"Available APIDs in data: {list(dataset_dict.keys())}")

    if not dataset_dict:
        logger.warning("No packets found in the provided files.")
        return pd.DataFrame()

    if apid not in dataset_dict:
        logger.error(f"APID {apid} not found in the dataset. Available APIDs: {list(dataset_dict.keys())}")
        return pd.DataFrame()

    dataset = dataset_dict[apid]
    raw_df = dataset.to_dataframe().reset_index()

    logger.info(f"Found {len(raw_df)} packets for APID {apid} ({config.name})")

    # Process based on whether this APID has samples or not
    source_file = packet_files[0].name if packet_files else "unknown"

    if not config.has_samples:
        parsed_data = process_non_sample_apid(raw_df, config, source_file, utc_offset_hours)
        logger.info(f"Extracted {len(parsed_data)} records from {len(raw_df)} packets")

    return pd.DataFrame(parsed_data)


def get_xtce_file(xtce_dir: Path | None = None) -> Path:
    """Get path to XTCE definition file."""
    if xtce_dir is None:
        xtce_dir = Path(__file__).parent.parent / "ground_calibration_data"

    xtce_file = xtce_dir / "icie_xtce_tlm.xml"

    if not xtce_file.exists():
        raise FileNotFoundError(f"XTCE file not found: {xtce_file}")

    return xtce_file


def get_list_ccsds_data_files(folder_path: Path = None) -> list[Path]:
    """Get list of packet files to analyze."""
    if folder_path is None:
        data_dir = Path(__file__).parent.parent / "binary_data"
    else:
        data_dir = folder_path

    working_files = []

    for file in sorted(data_dir.glob("ccsds_2025_*")):
        if file.is_file():
            working_files.append(file)

    packet_files = working_files

    logger.info(f"Using {len(packet_files)} packet file(s):")
    for file_path in packet_files:
        logger.info(f"  - {file_path.name}")

    return packet_files


def analyze_specific_apids(
    apids: list[int], packet_files: list[Path] | None = None, xtce_file: Path | None = None, utc_offset_hours=0
) -> dict[int, pd.DataFrame]:
    """
    Analyze specific APIDs with comprehensive reporting.

    Args:
        apids: List of APID numbers to analyze
        packet_files: Packet files to process (uses default if None)
        xtce_file: XTCE definition file (uses default if None)
        plot: Whether to create plots
        utc_offset_hours: UTC offset in hours (0 for UTC, -6 for MDT, etc.)

    Returns:
        Dictionary mapping APIDs to DataFrames
    """
    if packet_files is None:
        packet_files = get_list_ccsds_data_files()
    if xtce_file is None:
        xtce_file = get_xtce_file()

    logger.info(f"Analyzing APIDs: {apids}")

    # Parse the data
    results = parse_apid_packets(packet_files, xtce_file, apids, utc_offset_hours)

    return results


def select_out_image_blobs_from_binary_data(parsed_data: pd.DataFrame) -> pd.DataFrame:
    """Select image blobs from binary data packets."""
    image_start_idxs = np.where(parsed_data.mem_dump_flags == "SOP")[0]
    image_end_idxs = np.where(parsed_data.mem_dump_flags == "EOP")[0]
    image_blobs = []
    for i in range(len(image_end_idxs)):
        start_index = image_start_idxs[i]
        end_index = image_end_idxs[i]
        if end_index < start_index:
            end_index = image_end_idxs[i + 1]
        logger.info(f"Image start at index: {start_index}, Timestamp: {parsed_data.iloc[start_index]['TIMESTAMP']}")
        logger.info(f"Image end at index: {end_index}, Timestamp: {parsed_data.iloc[end_index]['TIMESTAMP']}")
        image_bytes = retrieve_image_bytes_from_wfov_dataframe(parsed_data, start_index, end_index)

        image_blobs.append(image_bytes)

    return image_blobs


def retrieve_image_bytes_from_wfov_dataframe(df, start_packet_index, end_packet_index):
    """
    Retrieve image bytes from a DataFrame containing WFOV data.
    """
    if start_packet_index < 0 or end_packet_index >= len(df):
        raise IndexError("Packet indices are out of bounds")

    if df["mem_dump_flags"][start_packet_index] != "SOP":
        raise ValueError("Start packet does not have SOP flag")
    if df["mem_dump_flags"][end_packet_index] != "EOP":
        raise ValueError("End packet does not have EOP flag")

    # Extract the relevant data
    # flags = df.mem_dump_flags[start_packet_index : end_packet_index + 1]
    image_data = df["science_data"][start_packet_index : end_packet_index + 1]
    packet_lengths = df["mem_dump_length"][start_packet_index : end_packet_index + 1]

    # Convert to bytearray by iterating through the series and using the packet lengths
    image_bytes = bytearray()
    for i, packet in enumerate(image_data):
        if i < len(packet_lengths):
            length = packet_lengths.iloc[i]
            if length > 0:
                image_bytes.extend(packet[:length])

    return image_bytes


def extract_compressed_image_data_from_bytearray(image_bytearray: bytearray) -> bytearray:
    """
    Extract compressed data from byte array, simulating the file reading logic
    """
    FPGA_FILE_HEADER_SIZE = 140
    FSW_FILE_HEADER_SIZE = 36
    FPGA_FILE_FOOTER_SIZE = 8

    # Total size of our byte data
    file_size = len(image_bytearray)

    # Compute size of compressed portion
    compressed_size = file_size - (FPGA_FILE_HEADER_SIZE + FSW_FILE_HEADER_SIZE) - FPGA_FILE_FOOTER_SIZE

    # Extract compressed data (skip headers, exclude footer)
    start_pos = FPGA_FILE_HEADER_SIZE + FSW_FILE_HEADER_SIZE
    end_pos = start_pos + compressed_size

    compressed_data = image_bytearray[start_pos:end_pos]

    return compressed_data


def process_all_available_images(apid1040_results: pd.DataFrame):
    """Process all available images from APID 1040 results."""
    image_bytes = select_out_image_blobs_from_binary_data(apid1040_results)
    metadatas = []
    images = []
    integration_time_masks = []
    for image in image_bytes:
        logger.info(f"Extracted image blob of size: {len(image)} bytes")
        pixel_data = extract_compressed_image_data_from_bytearray(image)
        img_metadata = extract_metadata(image, image_data=pixel_data)
        img_metadata["timestamp"] = tai_epoch_to_datetime(
            img_metadata["timestamp_seconds"], img_metadata["timestamp_subseconds"]
        )
        logger.info(f"Image timestamp: {img_metadata['timestamp']}")
        metadatas.append(img_metadata)
        image = Image.open(io.BytesIO(pixel_data))
        image_data = np.array(image)
        image_12bits = image_data & 0x0FFF
        images.append(image_12bits)
        image_13thbit = (image_data >> 12) & 0x0001
        integration_time_masks.append(image_13thbit)

    return images, metadatas, integration_time_masks


if __name__ == "__main__":
    # Example usage: analyze APID 1040

    # Read data files (default folder is the binary_data directory in libera_cam)
    data_files = get_list_ccsds_data_files()
    # Read binary data from packet files for APID 1040
    results = analyze_specific_apids(1040, packet_files=data_files)
    # Process images from APID 1040 results
    images, metadatas, integration_time_masks = process_all_available_images(results)

    logger.info("Done processing images.")
