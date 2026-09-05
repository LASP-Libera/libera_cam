"""Build a lazy WFOV working dataset from image-centric L1A NetCDF products."""

from __future__ import annotations

import logging
import os

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from libera_utils.l1a.wfov_image_metadata import (
    CAMERA_PACKET_INDEX_VAR,
    CAMERA_TIME_COORD,
    ERROR_FLAGGED_IMAGE_COUNT_ATTR,
    FIRST_IMAGE_INCOMPLETE_ATTR,
    FOOTER_MISMATCH_COUNT_ATTR,
    HEADER_PARSE_ERROR_COUNT_ATTR,
    LAST_IMAGE_INCOMPLETE_ATTR,
    PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR,
    WFOV_COMPRESSED_IMAGE_LENGTH_VAR,
    WFOV_COMPRESSED_IMAGE_VAR,
    WFOV_HEADER_PARSE_VALID_VAR,
)

from libera_cam import constants
from libera_cam.constants import DEFAULT_TIME_CHUNK_SIZE
from libera_cam.image_parsing import l1a_parser
from libera_cam.image_parsing.exposure import actual_exposure_counts_to_ms, delta_exposure_counts_to_ms

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_REQUIRED_L1A_VARS = (
    CAMERA_TIME_COORD,
    WFOV_COMPRESSED_IMAGE_VAR,
    WFOV_COMPRESSED_IMAGE_LENGTH_VAR,
)

# L1A variable -> internal working-dataset name used by packaging. libera_utils prefixes each
# decoded header field with its metadata block (WFOV_FSW_HEADER_, WFOV_IMAGE_HEADER_,
# WFOV_IMAGE_FOOTER_, WFOV_FPGA_STATUS_); see icie_wfov_sci_l1a.yml for the full field list.
_L1A_METADATA_MAP = {
    "WFOV_FSW_HEADER_AZIMUTH_ANGLE": "azimuth_angle",
    "WFOV_FSW_HEADER_RAD_OBS_ID": "rad_obs_id",
    "WFOV_FSW_HEADER_CAM_OBS_ID": "cam_obs_id",
    "WFOV_FSW_HEADER_IMG_MODE": "img_mode",
    CAMERA_PACKET_INDEX_VAR: "camera_packet_index",
}

_ACTUAL_EXP_TIME_1_VAR = "WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1"
_ACTUAL_EXP_TIME_2_VAR = "WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_2"
_DELTA_EXP_VAR = "WFOV_IMAGE_HEADER_DELTA"

_STITCH_QUALITY_ATTRS = (
    PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR,
    ERROR_FLAGGED_IMAGE_COUNT_ATTR,
    FOOTER_MISMATCH_COUNT_ATTR,
    HEADER_PARSE_ERROR_COUNT_ATTR,
    FIRST_IMAGE_INCOMPLETE_ATTR,
    LAST_IMAGE_INCOMPLETE_ATTR,
)


def _require_image_centric_l1a(cam_dataset: xr.Dataset) -> None:
    """Raise if the dataset is not an image-enhanced WFOV L1A product."""
    missing = [name for name in _REQUIRED_L1A_VARS if name not in cam_dataset and name not in cam_dataset.coords]
    if missing:
        raise ValueError(
            "L1A dataset is missing image-centric WFOV fields "
            f"{missing}. libera_cam requires libera-utils >= 5.10.5 L1A products with "
            f"{CAMERA_TIME_COORD}, {WFOV_COMPRESSED_IMAGE_VAR}, and {WFOV_COMPRESSED_IMAGE_LENGTH_VAR}."
        )


def _drop_unparsed_headers(cam_dataset: xr.Dataset) -> xr.Dataset:
    """Drop CAMERA_TIME rows whose FSW/FPGA header failed to decode at L1A.

    Rows flagged invalid carry a NaT ``CAMERA_TIME`` and zero-filled metadata, so they cannot be
    geolocated or attributed to an acquisition instant. Raises if no row survives.
    """
    if WFOV_HEADER_PARSE_VALID_VAR not in cam_dataset:
        return cam_dataset

    valid = np.asarray(cam_dataset[WFOV_HEADER_PARSE_VALID_VAR].values, dtype=bool)
    n_invalid = int((~valid).sum())
    if n_invalid == 0:
        return cam_dataset

    if not valid.any():
        raise ValueError(
            f"All {valid.size} L1A images have {WFOV_HEADER_PARSE_VALID_VAR} False; "
            "no image carries a usable CAMERA_TIME."
        )

    logger.warning(
        "Dropping %d of %d L1A images with %s False (NaT CAMERA_TIME, zero-filled metadata)",
        n_invalid,
        valid.size,
        WFOV_HEADER_PARSE_VALID_VAR,
    )
    return cam_dataset.isel({CAMERA_TIME_COORD: valid})


def _validate_execution_config(chunk_size: int) -> None:
    """Validate runtime chunk size and compare against worker memory limits if set."""
    if chunk_size < 1:
        raise ValueError(f"LIBERA_CAM_CHUNK_SIZE must be >= 1, got {chunk_size}")

    # Estimate ~3.3 GB per 50-image chunk (~67 MB per image for raw + float32 geolocation arrays)
    estimated_chunk_bytes = chunk_size * (constants.PIXEL_COUNT_Y * constants.PIXEL_COUNT_X * 4 * 4)
    raw_memory_limit = os.getenv("DASK_MEMORY_LIMIT")
    if raw_memory_limit:
        try:
            from dask.utils import parse_bytes

            memory_limit_bytes = parse_bytes(raw_memory_limit)
            if estimated_chunk_bytes > memory_limit_bytes:
                chunk_gb = estimated_chunk_bytes / (1024**3)
                limit_gb = memory_limit_bytes / (1024**3)
                logger.warning(
                    f"Configured LIBERA_CAM_CHUNK_SIZE={chunk_size} requires ~{chunk_gb:.2f} GB per chunk, "
                    f"which exceeds DASK_MEMORY_LIMIT={raw_memory_limit} (~{limit_gb:.2f} GB). "
                    "Workers may experience Out-Of-Memory errors."
                )
        except Exception as e:
            logger.debug("Could not parse DASK_MEMORY_LIMIT (%s): %s", raw_memory_limit, e)


def _extract_jpeg_ls_payloads(cam_dataset: xr.Dataset) -> list[bytes]:
    """Slice JPEG-LS bytes from ``WFOV_COMPRESSED_IMAGE`` using per-image lengths."""
    lengths = cam_dataset[WFOV_COMPRESSED_IMAGE_LENGTH_VAR].values
    images = cam_dataset[WFOV_COMPRESSED_IMAGE_VAR].values
    payloads: list[bytes] = []
    for i in range(len(lengths)):
        length = int(lengths[i])
        payloads.append(images[i, :length].tobytes())
    return payloads


def read_l1a_cam_data(cam_dataset: xr.Dataset) -> xr.Dataset:
    """Build a lazy WFOV camera dataset from image-centric L1A data.

    Reads complete compressed JPEG-LS payloads from ``WFOV_COMPRESSED_IMAGE``, decompresses them
    in batches controlled by ``LIBERA_CAM_CHUNK_SIZE``, and attaches FSW/FPGA
    metadata already present on the L1A ``CAMERA_TIME`` plane.

    Parameters
    ----------
    cam_dataset : xr.Dataset
        Image-enhanced L1A WFOV SCI DECODED dataset (libera-utils >= 5.10.5).

    Returns
    -------
    xr.Dataset
        Lazy dataset with ``image_data``, ``integration_mask``, per-frame metadata
        on ``camera_time``, and derived flags ``good_image_flag`` and
        ``valid_pixel_mask``. Returns an empty dataset when no complete images
        are found.
    """
    _require_image_centric_l1a(cam_dataset)
    cam_dataset = _drop_unparsed_headers(cam_dataset)

    chunk_size = int(os.getenv("LIBERA_CAM_CHUNK_SIZE", DEFAULT_TIME_CHUNK_SIZE))
    _validate_execution_config(chunk_size)
    logger.info("Using LIBERA_CAM_CHUNK_SIZE=%s", chunk_size)

    n_images = int(cam_dataset.sizes[CAMERA_TIME_COORD])
    if n_images == 0:
        logger.warning("No complete images found in L1A data.")
        return xr.Dataset()

    def decompress_batch(payloads_batch: list[bytes]):
        """Decompress a list of JPEG-LS payloads and return stacked arrays."""
        images, masks = [], []
        for payload in payloads_batch:
            img, mask = l1a_parser.decompress_image(payload)
            images.append(img)
            masks.append(mask)
        return np.stack(images, axis=0), np.stack(masks, axis=0)

    delayed_decompress_batch = delayed(decompress_batch, nout=2, pure=False)

    logger.info("Extracting JPEG-LS payloads from L1A WFOV_COMPRESSED_IMAGE...")
    payloads = _extract_jpeg_ls_payloads(cam_dataset)
    times = np.asarray(cam_dataset[CAMERA_TIME_COORD].values)
    logger.info("Found %s complete images. Constructing Dask graph...", n_images)

    delayed_image_chunks, delayed_mask_chunks = [], []
    for i in range(0, n_images, chunk_size):
        batch = payloads[i : min(n_images, i + chunk_size)]
        actual_size = len(batch)
        out = delayed_decompress_batch(batch)
        delayed_image_chunks.append(
            da.from_delayed(
                out[0], shape=(actual_size, constants.PIXEL_COUNT_Y, constants.PIXEL_COUNT_X), dtype=np.int32
            )
        )
        delayed_mask_chunks.append(
            da.from_delayed(
                out[1], shape=(actual_size, constants.PIXEL_COUNT_Y, constants.PIXEL_COUNT_X), dtype=np.uint8
            )
        )

    image_data_3d = da.concatenate(delayed_image_chunks, axis=0)
    integration_mask_3d = da.concatenate(delayed_mask_chunks, axis=0)

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

    # The L1A product definition makes every one of these mandatory, so a missing name means a
    # product-version mismatch rather than an absent optional field.
    expected = (*_L1A_METADATA_MAP, _ACTUAL_EXP_TIME_1_VAR, _ACTUAL_EXP_TIME_2_VAR, _DELTA_EXP_VAR)
    missing_metadata = [name for name in expected if name not in cam_dataset]
    if missing_metadata:
        raise ValueError(f"L1A dataset is missing expected WFOV header metadata variables: {missing_metadata}")

    for l1a_name, internal_name in _L1A_METADATA_MAP.items():
        ds[internal_name] = (("camera_time",), np.asarray(cam_dataset[l1a_name].values))

    ds["actual_exposure_time_1"] = (
        ("camera_time",),
        actual_exposure_counts_to_ms(cam_dataset[_ACTUAL_EXP_TIME_1_VAR].values),
    )
    ds["actual_exposure_time_2"] = (
        ("camera_time",),
        actual_exposure_counts_to_ms(cam_dataset[_ACTUAL_EXP_TIME_2_VAR].values),
    )
    ds["exposure_delta"] = (
        ("camera_time",),
        delta_exposure_counts_to_ms(cam_dataset[_DELTA_EXP_VAR].values),
    )

    ds["good_image_flag"] = ds["image_data"].max(dim=["y", "x"]) > 0
    ds["valid_pixel_mask"] = ds["image_data"] > 0
    ds["valid_pixel_mask"].attrs = {
        "long_name": "Valid Pixel Mask",
        "description": "True where image_data > 0, False otherwise. Used to mask geolocation.",
    }

    ds.attrs["description"] = "WFOV Camera Image Cube reconstructed from L1A image blobs"
    ds.attrs["n_images_decoded"] = n_images
    for attr_name in _STITCH_QUALITY_ATTRS:
        if attr_name in cam_dataset.attrs:
            ds.attrs[attr_name] = cam_dataset.attrs[attr_name]

    logger.info("Constructed lazy Dataset with %s time steps.", n_images)
    return ds
