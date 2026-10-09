"""Integration tests for reading L1A camera data from DITL files."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data

# TODO[LIBSDC-844]: This needs to be updated/confirmed and documented
_DUAL_EXPOSURE_LAG_MS = (111.0, 350.0)

# Reference values from decoding the DITL fixture's three WFOV_COMPRESSED_IMAGE payloads with
# pillow-jpls, the decoder libera_cam used before pyjpegls: pillow-jpls 1.3.2 built from its sdist
# (CharLS 2.4.2 via Conan 2.0.17), Pillow 12.3.0, Python 3.13, decoded as
# np.array(Image.open(payload), dtype=np.int32) and split into 12-bit DN and bit-12 mask.
# The PyPI cp312 wheel of pillow-jpls 1.3.2 decodes the same payloads to identical arrays, and both
# matched jpeg_ls.jlsread pixel for pixel. pillow-jpls is deliberately not a test dependency.
_PILLOW_JPLS_FRAME_SUMS = [  # (image_data sum, integration_mask sum) per camera_time row
    (244_254_807, 302_923),
    (2_068_130_237, 4_194_304),
    (250_540_867, 310_474),
]
_PILLOW_JPLS_FRAME1_PIXELS = [  # (y, x, image_data, integration_mask); frames 0 and 2 are mostly zero
    (0, 0, 192, 1),
    (512, 1536, 756, 1),
    (1024, 1024, 333, 1),
    (1536, 512, 380, 1),
    (2047, 2047, 200, 1),
]


@pytest.mark.integration
def test_read_ditl_l1a_cam_data(test_ditl_l1a_file_path):
    l1a_dataset = xr.open_dataset(test_ditl_l1a_file_path)
    image_dataset = read_l1a_cam_data(l1a_dataset)

    assert isinstance(image_dataset.image_data.data, da.Array), "Image data should be a Dask array"
    assert isinstance(image_dataset.integration_mask.data, da.Array), "Integration mask should be a Dask array"

    assert len(image_dataset.camera_time) == 3
    assert image_dataset.image_data.shape == (3, 2048, 2048)
    assert image_dataset.integration_mask.shape == (3, 2048, 2048)

    assert image_dataset.good_image_flag.sum().values == 3

    assert "valid_pixel_mask" in image_dataset
    assert isinstance(image_dataset.valid_pixel_mask.data, da.Array), "Valid pixel mask should be a Dask array"

    img_slice = image_dataset.image_data.isel(camera_time=0).compute()
    mask_slice = image_dataset.valid_pixel_mask.isel(camera_time=0).compute()

    expected_mask = img_slice > 0
    np.testing.assert_array_equal(mask_slice, expected_mask)

    # Decoded values match the pillow-jpls decoder this package used before pyjpegls
    image_data = image_dataset.image_data.values
    integration_mask = image_dataset.integration_mask.values
    frame_sums = [
        (int(image_data[i].sum(dtype=np.int64)), int(integration_mask[i].sum(dtype=np.int64)))
        for i in range(len(_PILLOW_JPLS_FRAME_SUMS))
    ]
    assert frame_sums == _PILLOW_JPLS_FRAME_SUMS
    for y, x, dn, mask in _PILLOW_JPLS_FRAME1_PIXELS:
        assert image_data[1, y, x] == dn
        assert integration_mask[1, y, x] == mask


@pytest.mark.integration
def test_read_ditl_l1a_header_metadata_values(test_ditl_l1a_file_path):
    """Header metadata carries the DITL fixture's actual FSW values, not just the variable names."""
    image_dataset = read_l1a_cam_data(xr.open_dataset(test_ditl_l1a_file_path))

    np.testing.assert_array_equal(image_dataset["img_mode"].values, [1, 1, 1])
    np.testing.assert_array_equal(image_dataset["rad_obs_id"].values, [132, 132, 132])
    np.testing.assert_array_equal(image_dataset["cam_obs_id"].values, [133, 133, 132])
    np.testing.assert_array_equal(image_dataset["camera_packet_index"].values, [1001, 1301, 4433])
    np.testing.assert_allclose(image_dataset["azimuth_angle"].values, [-3.9478302, -3.9478302, -6.4626794], rtol=1e-6)

    # The fixture is a VIDEO-mode granule, so the first two images share one FSW timestamp and are
    # told apart only by camera_packet_index.
    assert image_dataset["camera_time"].values[0] == image_dataset["camera_time"].values[1]


@pytest.mark.integration
def test_ditl_exposure_conversions_match_commanded_times(test_ditl_l1a_file_path):
    """Converted FPGA exposure registers agree with the independently-recorded commanded times.

    ``WFOV_FSW_HEADER_COMMANDED_EXP_TIME_*`` is already milliseconds at L1A and is decoded from the
    FSW header block, while the actual exposures come from the FPGA image header block. Agreement
    between them is what pins the conversion constants; a test that recomputed the conversion
    equation would pass with any constants.
    """
    l1a_dataset = xr.open_dataset(test_ditl_l1a_file_path)
    image_dataset = read_l1a_cam_data(l1a_dataset)

    np.testing.assert_allclose(
        image_dataset["actual_exposure_time_1"].values,
        l1a_dataset["WFOV_FSW_HEADER_COMMANDED_EXP_TIME_1"].values,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        image_dataset["actual_exposure_time_2"].values,
        l1a_dataset["WFOV_FSW_HEADER_COMMANDED_EXP_TIME_2"].values,
        rtol=1e-3,
    )

    # TODO[LIBSDC-844]: This needs to be confirmed and documented better
    delta_ms = image_dataset["exposure_delta"].values
    low, high = _DUAL_EXPOSURE_LAG_MS
    assert np.all((delta_ms >= low) & (delta_ms <= high)), delta_ms
