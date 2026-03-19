import logging
from collections.abc import Generator
from datetime import datetime

import dask.array as da
import dask.delayed
import numpy as np
import xarray as xr
from libera_utils.time import multipart_to_dt64

# Import the new separated parser functions
from libera_cam import constants
from libera_cam.image_parsing import l1a_parser

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def reassemble_image_blobs(l1a_data: xr.Dataset, stats: dict = None) -> Generator[bytearray, None, None]:
    """
    Generator that iterates through the CCSDS packet stream to stitch together image blobs.

    Uses ICIE__MEM_DUMP_FLAGS_WFOV to find SOP (Start of Packet) and EOP (End of Packet).
    Yields image blobs one by one to conserve memory.

    Parameters
    ----------
    l1a_data : xr.Dataset
        Input L1A dataset.
    stats : dict, optional
        Dictionary to track stitching statistics in-place.
    """
    if stats is None:
        # Initialize a fresh stats dictionary with all counters set to zero.
        stats = {
            "n_packets_read": 0,
            "n_images_stitched": 0,
            "n_images_discarded_sop": 0,
            "n_images_discarded_gap": 0,
            "n_unexpected_eop": 0,
        }
    else:
        # Only initialize missing keys so existing counts can accumulate across calls.
        stats.setdefault("n_packets_read", 0)
        stats.setdefault("n_images_stitched", 0)
        stats.setdefault("n_images_discarded_sop", 0)
        stats.setdefault("n_images_discarded_gap", 0)
        stats.setdefault("n_unexpected_eop", 0)

    offsets = l1a_data["ICIE__MEM_DUMP_OFFSET_WFOV"].values
    lengths = l1a_data["ICIE__MEM_DUMP_LENGTH_WFOV"].values
    flags = l1a_data["ICIE__MEM_DUMP_FLAGS_WFOV"].values.astype(str)
    all_packet_data = l1a_data["ICIE__WFOV_DATA"].data

    state = "SEEKING"
    start_index = -1
    expected_offset = 0

    for i in range(len(flags)):
        stats["n_packets_read"] += 1
        flag = flags[i]
        offset = offsets[i]
        length = lengths[i]

        if flag == "SOP":
            if state == "COLLECTING":
                logger.warning(f"Unexpected SOP at index {i} while collecting image. Discarding partial image.")
                stats["n_images_discarded_sop"] += 1

            state = "COLLECTING"
            start_index = i
            if offset != 0:
                logger.warning(f"SOP at index {i} has non-zero offset {offset}. Discarding.")
                stats["n_images_discarded_sop"] += 1
                state = "SEEKING"
                continue
            expected_offset = length

        elif state == "COLLECTING":
            # Check for offset continuity
            if offset != expected_offset:
                logger.warning(
                    f"Offset discontinuity at index {i}. Expected {expected_offset}, got {offset}. "
                    "Discarding partial image."
                )
                stats["n_images_discarded_gap"] += 1
                state = "SEEKING"
                continue

            expected_offset += length

            if flag == "EOP":
                # Efficiently compute the whole image blob at once
                # This minimizes Dask overhead by performing one compute per image
                image_packets = all_packet_data[start_index : i + 1]

                if hasattr(image_packets, "compute"):
                    image_packets = image_packets.compute()

                # Reconstruct blob from packets and their actual lengths
                # We view as uint8 to ensure we get the full buffer even if it contains nulls
                # (Numpy fixed-length strings |S972 would otherwise strip trailing nulls)
                packet_width = all_packet_data.dtype.itemsize
                image_packets_raw = image_packets.view(np.uint8).reshape(-1, packet_width)
                packet_lengths = lengths[start_index : i + 1]
                parts = []
                for p_idx in range(len(image_packets_raw)):
                    p_len = packet_lengths[p_idx]
                    packet_data = image_packets_raw[p_idx]
                    parts.append(packet_data[:p_len].tobytes())

                blob = bytearray(b"".join(parts))
                stats["n_images_stitched"] += 1
                yield blob

                state = "SEEKING"
        else:
            # SEEKING and not SOP
            if flag == "EOP":
                logger.warning(f"Unexpected EOP at index {i} while seeking SOP. Ignoring.")
                stats["n_unexpected_eop"] += 1
            continue


def read_l1a_cam_data(cam_dataset: xr.Dataset) -> xr.Dataset:
    """
    Reads a Libera CAM L1A data product and returns a stacked Image Cube backed by Dask Arrays.

    Parameters
    ----------
    cam_dataset : xr.Dataset
        Input L1A dataset containing the stream of CCSDS packets.

    Returns
    -------
    xr.Dataset
        A single Xarray Dataset containing:
        - 3D variables (camera_time, y, x): image_data, integration_mask (Lazy Dask Arrays)
        - 1D variables (camera_time): gain, exposure_time, temperature, etc. (Eager Loaded)
        - Coordinates: camera_time, y, x
    """
    # 1. Extract raw binary blobs from the packet stream
    logger.info("Stitching binary blobs from L1A packets...")
    start = datetime.now()

    # Process Blobs lazily
    metadata_list = []
    delayed_image_chunks = []
    delayed_mask_chunks = []
    stitching_stats = {}

    # Use a dummy wrapper to handle the tuple return from decompress_image
    # This allows us to index into the delayed object to get the two separate arrays
    delayed_decompress = dask.delayed(l1a_parser.decompress_image, nout=2)

    for blob in reassemble_image_blobs(cam_dataset, stats=stitching_stats):
        try:
            # Eager Metadata Parse
            meta = l1a_parser.parse_image_metadata(blob)

            # Compute timestamp for coordination
            meta["camera_time"] = multipart_to_dt64(meta, s_field="timestamp_seconds", us_field="timestamp_subseconds")
            metadata_list.append(meta)

            # Lazy Image Decompression
            # Returns tuple (image, mask)
            out = delayed_decompress(blob)

            # We must specify shape and dtype for from_delayed to work without computing
            image_chunk = da.from_delayed(
                out[0], shape=(constants.PIXEL_COUNT_Y, constants.PIXEL_COUNT_X), dtype=np.int32
            )
            mask_chunk = da.from_delayed(
                out[1], shape=(constants.PIXEL_COUNT_Y, constants.PIXEL_COUNT_X), dtype=np.uint8
            )

            delayed_image_chunks.append(image_chunk)
            delayed_mask_chunks.append(mask_chunk)

        except Exception as e:
            logger.exception(f"Failed to process image blob: {e}")
            continue

    end = datetime.now()
    logger.info(f"Image data extraction and metadata parsing in {(end - start).total_seconds()} seconds")

    if not metadata_list:
        logger.warning("No complete images found in L1A data.")
        return xr.Dataset()

    logger.info(f"Found {len(metadata_list)} valid images. Constructing Dask Graph...")

    # 3. Stack into 3D Dask Arrays
    # Stack along the time dimension (axis 0)
    image_data_3d = da.stack(delayed_image_chunks, axis=0)
    integration_mask_3d = da.stack(delayed_mask_chunks, axis=0)

    # 4. Construct Coordinate Arrays
    times = [m["camera_time"] for m in metadata_list]

    # Consolidate metadata into a dict of lists
    consolidated_meta = {}
    if metadata_list:
        keys = metadata_list[0].keys()
        for key in keys:
            if key not in ["camera_time", "timestamp_seconds", "timestamp_subseconds"]:
                consolidated_meta[key] = [m.get(key) for m in metadata_list]

    # 5. Create Dataset
    ds = xr.Dataset(
        data_vars={
            "image_data": (("camera_time", "y", "x"), image_data_3d),
            "integration_mask": (("camera_time", "y", "x"), integration_mask_3d),
        },
        coords={
            "camera_time": times,
            "y": np.arange(constants.PIXEL_COUNT_Y),
            "x": np.arange(constants.PIXEL_COUNT_X),
        },
    )

    # Add 1D metadata variables
    for key, values in consolidated_meta.items():
        ds[key] = (("camera_time",), values)

    # Add 'good_image_flag' logic (Lazy)
    # This must be lazy because it depends on the image data!
    # Original logic: is_good = np.max(image_12bit) > 0
    # Dask supports .max(), so this operation will remain lazy
    ds["good_image_flag"] = ds["image_data"].max(dim=["y", "x"]) > 0

    # Add 'valid_pixel_mask' (Lazy)
    # Used to skip geolocation for dark/invalid pixels (e.g. fisheye corners)
    # Logic: Any pixel with value > 0 is considered potentially valid for geolocation.
    ds["valid_pixel_mask"] = ds["image_data"] > 0
    ds["valid_pixel_mask"].attrs = {
        "long_name": "Valid Pixel Mask",
        "description": "True where image_data > 0, False otherwise. Used to mask geolocation.",
    }

    ds.attrs["description"] = "WFOV Camera Image Cube reconstructed from L1A packets"
    # Attach stitching statistics
    for key, val in stitching_stats.items():
        ds.attrs[key] = val
    ds.attrs["n_images_decoded"] = len(metadata_list)

    logger.info(f"Constructed lazy Dataset with {len(times)} time steps.")

    return ds
