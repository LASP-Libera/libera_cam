"""Unit tests for building the working dataset from image-centric L1A products."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from libera_cam import constants
from libera_cam.image_parsing.read_l1a_cam_data import _validate_execution_config, read_l1a_cam_data

_PAYLOAD_LENGTH = 8


def make_synthetic_l1a(n_images: int = 3, valid: list[bool] | None = None, drop: tuple[str, ...] = ()) -> xr.Dataset:
    """Build a minimal image-centric L1A dataset (libera-utils >= 5.10.5 layout).

    Each image's payload bytes are filled with its own index + 1, so a decompressed image can be
    traced back to the L1A row it came from.

    Parameters
    ----------
    n_images : int
        Number of CAMERA_TIME rows.
    valid : list[bool] or None
        Per-image ``WFOV_HEADER_PARSE_VALID``. Defaults to all True.
    drop : tuple[str, ...]
        Variable names to omit, for testing the missing-field errors.
    """
    valid_flags = np.ones(n_images, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    times = np.array(["2028-02-12T03:41:36"], dtype="datetime64[ns]") + np.arange(n_images) * np.timedelta64(1, "s")

    # BLOB_BYTE is padded past the valid length, as libera_utils writes it.
    blob = np.zeros((n_images, _PAYLOAD_LENGTH * 2), dtype=np.uint8)
    for i in range(n_images):
        blob[i, :_PAYLOAD_LENGTH] = i + 1

    data_vars = {
        "WFOV_COMPRESSED_IMAGE": (("CAMERA_TIME", "BLOB_BYTE"), blob),
        "WFOV_COMPRESSED_IMAGE_LENGTH": (("CAMERA_TIME",), np.full(n_images, _PAYLOAD_LENGTH, dtype=np.uint32)),
        "WFOV_HEADER_PARSE_VALID": (("CAMERA_TIME",), valid_flags),
        "WFOV_FSW_HEADER_AZIMUTH_ANGLE": (("CAMERA_TIME",), np.arange(n_images, dtype=np.float32)),
        "WFOV_FSW_HEADER_RAD_OBS_ID": (("CAMERA_TIME",), np.full(n_images, 132, dtype=np.uint16)),
        "WFOV_FSW_HEADER_CAM_OBS_ID": (("CAMERA_TIME",), np.full(n_images, 133, dtype=np.uint16)),
        "WFOV_FSW_HEADER_IMG_MODE": (("CAMERA_TIME",), np.ones(n_images, dtype=np.uint8)),
        "CAMERA_PACKET_INDEX": (("CAMERA_TIME",), np.arange(n_images, dtype=np.int32)),
        "WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1": (("CAMERA_TIME",), np.full(n_images, 41, dtype=np.uint32)),
        "WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_2": (("CAMERA_TIME",), np.full(n_images, 1232, dtype=np.uint32)),
        "WFOV_IMAGE_HEADER_DELTA": (("CAMERA_TIME",), np.full(n_images, 2240000, dtype=np.uint32)),
    }
    for name in drop:
        del data_vars[name]

    return xr.Dataset(data_vars, coords={"CAMERA_TIME": times})


def _fake_decompress(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Stand in for JPEG-LS decode, returning an image filled with the payload's marker byte."""
    shape = (constants.PIXEL_COUNT_Y, constants.PIXEL_COUNT_X)
    return np.full(shape, payload[0], dtype=np.int32), np.zeros(shape, dtype=np.uint8)


def test_drops_invalid_header_rows_keeping_images_aligned_with_metadata():
    """Rows flagged WFOV_HEADER_PARSE_VALID False are dropped from images and metadata together."""
    l1a = make_synthetic_l1a(n_images=4, valid=[True, False, False, True])

    with patch("libera_cam.image_parsing.l1a_parser.decompress_image", side_effect=_fake_decompress):
        ds = read_l1a_cam_data(l1a)
        image_markers = ds["image_data"][:, 0, 0].compute().values

    assert ds.sizes["camera_time"] == 2
    # Rows 0 and 3 survive: their metadata, times, and image payloads must all still line up.
    np.testing.assert_array_equal(ds["camera_packet_index"].values, [0, 3])
    np.testing.assert_array_equal(ds["azimuth_angle"].values, [0.0, 3.0])
    np.testing.assert_array_equal(ds["camera_time"].values, l1a["CAMERA_TIME"].values[[0, 3]])
    np.testing.assert_array_equal(image_markers, [1, 4])


def test_all_invalid_headers_raises():
    """An L1A granule with no usable CAMERA_TIME fails loudly rather than producing an empty product."""
    with pytest.raises(ValueError, match="WFOV_HEADER_PARSE_VALID"):
        read_l1a_cam_data(make_synthetic_l1a(n_images=2, valid=[False, False]))


def test_missing_header_metadata_raises_naming_the_variable():
    """A product-version mismatch names the absent variable instead of silently dropping it."""
    l1a = make_synthetic_l1a(drop=("WFOV_IMAGE_HEADER_DELTA",))

    with pytest.raises(ValueError, match="WFOV_IMAGE_HEADER_DELTA"):
        read_l1a_cam_data(l1a)


def test_rejects_packet_only_l1a():
    """Packet-only L1A (pre-5.10.5) must raise a clear error."""
    ds = xr.Dataset({"ICIE__WFOV_DATA": (("PACKET",), np.zeros(4, dtype="|S972"))})

    with pytest.raises(ValueError, match="image-centric"):
        read_l1a_cam_data(ds)


def test_validate_execution_config():
    """Chunk size below 1 is rejected."""
    with pytest.raises(ValueError, match="LIBERA_CAM_CHUNK_SIZE must be >= 1"):
        _validate_execution_config(0)

    with pytest.raises(ValueError, match="LIBERA_CAM_CHUNK_SIZE must be >= 1"):
        _validate_execution_config(-5)

    _validate_execution_config(10)
