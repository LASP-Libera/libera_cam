import dask.array as da
import numpy as np
import xarray as xr

from libera_cam.image_parsing.read_l1a_cam_data import reassemble_image_blobs


def create_mock_l1a_dataset(flags, offsets, lengths, packet_data=None):
    """Helper to create a mock L1A dataset."""
    num_packets = len(flags)
    if packet_data is None:
        # Create dummy packet data (972 bytes per packet as fixed-length strings)
        # Use non-null bytes to avoid numpy's automatic truncation of null-filled strings
        packet_data = np.full(num_packets, b"A" * 972, dtype="S972")

    ds = xr.Dataset(
        data_vars={
            "ICIE__MEM_DUMP_FLAGS_WFOV": (("packet",), np.array(flags, dtype="S3")),
            "ICIE__MEM_DUMP_OFFSET_WFOV": (("packet",), np.array(offsets)),
            "ICIE__MEM_DUMP_LENGTH_WFOV": (("packet",), np.array(lengths)),
            "ICIE__WFOV_DATA": (("packet",), packet_data),
        }
    )
    return ds


def test_reassemble_image_blobs_clean_run():
    # 2 images, 3 packets each
    flags = ["SOP", "MOP", "EOP", "SOP", "MOP", "EOP"]
    offsets = [0, 100, 200, 0, 150, 300]
    lengths = [100, 100, 100, 150, 150, 150]

    ds = create_mock_l1a_dataset(flags, offsets, lengths)
    stats = {}
    blobs = list(reassemble_image_blobs(ds, stats=stats))

    assert len(blobs) == 2
    assert stats["n_images_stitched"] == 2
    assert stats["n_packets_read"] == 6
    assert len(blobs[0]) == 300
    assert len(blobs[1]) == 450


def test_reassemble_image_blobs_missing_eop():
    # First image is missing EOP, second is clean
    flags = ["SOP", "MOP", "SOP", "MOP", "EOP"]
    offsets = [0, 100, 0, 100, 200]
    lengths = [100, 100, 100, 100, 100]

    ds = create_mock_l1a_dataset(flags, offsets, lengths)
    stats = {}
    blobs = list(reassemble_image_blobs(ds, stats=stats))

    # Should only recover the second image
    assert len(blobs) == 1
    assert stats["n_images_stitched"] == 1
    assert stats["n_images_discarded_sop"] == 1
    assert len(blobs[0]) == 300


def test_reassemble_image_blobs_unexpected_eop():
    # EOP before any SOP
    flags = ["EOP", "SOP", "EOP"]
    offsets = [0, 0, 100]
    lengths = [100, 100, 100]

    ds = create_mock_l1a_dataset(flags, offsets, lengths)
    stats = {}
    blobs = list(reassemble_image_blobs(ds, stats=stats))

    assert len(blobs) == 1
    assert stats["n_unexpected_eop"] == 1
    assert len(blobs[0]) == 200


def test_reassemble_image_blobs_offset_discontinuity():
    # Offset jump in middle of image
    flags = ["SOP", "MOP", "EOP"]
    offsets = [0, 100, 300]  # Jump from 200 to 300
    lengths = [100, 100, 100]

    ds = create_mock_l1a_dataset(flags, offsets, lengths)
    stats = {}
    blobs = list(reassemble_image_blobs(ds, stats=stats))

    assert len(blobs) == 0
    assert stats["n_images_discarded_gap"] == 1


def test_reassemble_image_blobs_non_zero_sop():
    # SOP with non-zero offset
    flags = ["SOP", "EOP"]
    offsets = [10, 110]
    lengths = [100, 100]

    ds = create_mock_l1a_dataset(flags, offsets, lengths)
    stats = {}
    blobs = list(reassemble_image_blobs(ds, stats=stats))

    assert len(blobs) == 0
    assert stats["n_images_discarded_sop"] == 1


def test_reassemble_image_blobs_dask_backed():
    """Verify the generator works when the input WFOV_DATA is a Dask array."""
    flags = ["SOP", "EOP"]
    offsets = [0, 100]
    lengths = [100, 100]

    # Create dummy packet data as a Dask array
    # 2 packets, each 972 bytes
    raw_data = np.full((2,), b"D" * 972, dtype="S972")
    dask_data = da.from_array(raw_data, chunks=1)

    ds = create_mock_l1a_dataset(flags, offsets, lengths, packet_data=dask_data)

    stats = {}
    blobs = list(reassemble_image_blobs(ds, stats=stats))

    assert len(blobs) == 1
    # Check if content is correct (stitching worked across Dask chunks)
    # Each packet contributes 'lengths' bytes to the blob, so total should be 200 bytes of 'D'
    assert blobs[0] == bytearray(b"D" * 200)
    assert stats["n_images_stitched"] == 1
