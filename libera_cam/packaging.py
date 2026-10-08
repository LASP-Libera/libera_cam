"""
Module for packaging the L1B CAM product to ensure conformance with the product definition.
"""

import logging

import dask.array as da
import numpy as np
import xarray as xr
from libera_utils.io.product_definition import LiberaDataProductDefinition

from libera_cam.config import product_config_path

logger = logging.getLogger(__name__)

# Per-pixel product variables with no producer yet. Each is written as its product ``_FillValue``
# so the file says "not computed" rather than carrying zeros that look like data. Remove a name
# here when processing starts producing it: the strict writer then raises if it goes missing, and
# packaging raises if it arrives while still listed here, so neither direction fails silently.
# TODO[LIBSDC-814]: terrain-corrected coordinates.
_UNIMPLEMENTED_PIXEL_VARIABLES: tuple[str, ...] = (
    "Terrain_Corrected_Latitude",
    "Terrain_Corrected_Longitude",
    "Terrain_Corrected_Altitude",
)


def _placeholder_fill_values(names: tuple[str, ...]) -> dict[str, float]:
    """Product ``_FillValue`` for each named variable, read from the bundled product definition.

    Raises
    ------
    ValueError
        If a name is not a product variable or declares no ``_FillValue``.
    """
    definition = LiberaDataProductDefinition.from_yaml(product_config_path)
    fills = {}
    for name in names:
        if name not in definition.variables:
            raise ValueError(f"{name} is not a variable in {product_config_path.name}")
        attributes = definition.variables[name].attributes
        if "_FillValue" not in attributes:
            raise ValueError(f"{name} declares no _FillValue to use as its placeholder")
        fills[name] = attributes["_FillValue"]
    return fills


def package_l1b_product(dataset: xr.Dataset) -> xr.Dataset:
    """
    Refines the internal processing dataset into the final L1B product structure.

    Responsibilities:
    1. Rename internal dimensions and variables to match the L1B Product Definition.
    2. Transpose dimensions to match the required order (Time, X, Y).
    3. Create and assign placeholder variables for fields not yet computed.
    4. Enforce data types (float32, uint8, etc.).

    ``valid_pixel_mask``, the pixels the camera's onboard pixel mask kept, is written as
    ``Camera_Mask`` (1 kept, 0 removed); ``Radiance`` and the geolocation fields are fill where
    it is 0.

    Parameters
    ----------
    dataset : xr.Dataset
        The intermediate dataset containing science results (Radiance, Geolocation).

    Returns
    -------
    xr.Dataset
        The packaged dataset ready for NetCDF writing.

    Raises
    ------
    ValueError
        If the dataset lacks the FSW header ``azimuth_angle`` or the ``valid_pixel_mask`` that
        ``read_l1a_cam_data`` always provides, lacks ``Radiance``, or a placeholder variable is
        missing from the product definition or declares no ``_FillValue``.
    """
    logger.info("Packaging L1B product for conformance.")

    # Drop ingest-only summary attrs that are not part of the L1B product schema.
    for attr_name in (
        "description",
        "n_images_decoded",
        "n_packets_read",
        "n_images_stitched",
        "n_images_discarded_sop",
        "n_images_discarded_gap",
        "n_unexpected_eop",
        "n_images_failed_parse",
        "PacketCountNotUsedInImages",
        "ErrorFlaggedImageCount",
        "FooterMismatchCount",
        "HeaderParseErrorCount",
        "FirstImageIncomplete",
        "LastImageIncomplete",
    ):
        dataset.attrs.pop(attr_name, None)

    # ``Azimuth`` is the motor encoder angle from the SPICE CK, set during processing. The FSW
    # image-header ``azimuth_angle`` is a different quantity and passes through as
    # ``Azimuth_FSW_Metadata``.
    if "azimuth_angle" not in dataset:
        raise ValueError("Dataset must contain the FSW header 'azimuth_angle' variable from read_l1a_cam_data.")
    if "valid_pixel_mask" not in dataset:
        raise ValueError("Dataset must contain the 'valid_pixel_mask' variable from read_l1a_cam_data.")

    # 1. Rename variables/dims to match Product Definition.
    rename_map = {
        "azimuth_angle": "Azimuth_FSW_Metadata",
        "rad_obs_id": "Radiometer_Observation_ID",
        "cam_obs_id": "Camera_Observation_ID",
        "img_mode": "Image_Mode",
        "camera_packet_index": "Camera_Packet_Index",
        "actual_exposure_time_1": "Actual_Exposure_Time_1",
        "actual_exposure_time_2": "Actual_Exposure_Time_2",
        "exposure_delta": "Exposure_Delta",
        "image_data": "Pixel_Counts",
        "integration_mask": "Integration_Time_Flag",
        "good_image_flag": "Quality_Flag",
        "valid_pixel_mask": "Camera_Mask",
        "camera_time": "CAMERA_TIME",
        "x": "CAMERA_PIXEL_COUNT_X",
        "y": "CAMERA_PIXEL_COUNT_Y",
    }
    dataset = dataset.rename({src: dst for src, dst in rename_map.items() if src in dataset})
    # The ingest attributes describe valid_pixel_mask; the writer applies Camera_Mask's own.
    dataset["Camera_Mask"].attrs = {}

    # 2. Reorder dimensions to match product definition: (Time, X, Y)
    # Remaining dimensions (EUCLIDEAN_DIM on the spacecraft state vectors) keep their trailing position.
    dataset = dataset.transpose("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y", ...)

    # 3. Placeholders for the per-pixel fields without a producer (lazy, chunked like Radiance,
    # which is now (Time, X, Y)). Each carries its product _FillValue.
    if "Radiance" not in dataset:
        raise ValueError("Dataset must contain 'Radiance' variable before packaging.")

    radiance = dataset["Radiance"].data
    dims_3d = ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
    if produced := [name for name in _UNIMPLEMENTED_PIXEL_VARIABLES if name in dataset]:
        raise ValueError(
            f"{produced} carry data but are listed in _UNIMPLEMENTED_PIXEL_VARIABLES, which would "
            "overwrite them with fill values. Remove them from that list now that they have a producer."
        )
    for name, fill in _placeholder_fill_values(_UNIMPLEMENTED_PIXEL_VARIABLES).items():
        dataset[name] = (dims_3d, da.full_like(radiance, fill, dtype=np.float32))

    # 4. Ensure Types (Cast if necessary)
    # Using explicit casting to float32/uint types. The geolocation variables arrive typed from
    # their producer and carry no attributes; the writer applies the product definition's.
    type_map = {
        "Azimuth_FSW_Metadata": np.float32,
        "Radiometer_Observation_ID": np.uint16,
        "Camera_Observation_ID": np.uint16,
        "Image_Mode": np.uint8,
        "Camera_Packet_Index": np.int32,
        "Actual_Exposure_Time_1": np.float32,
        "Actual_Exposure_Time_2": np.float32,
        "Exposure_Delta": np.float32,
        "Pixel_Counts": np.uint16,
        "Integration_Time_Flag": np.uint8,
        "Camera_Mask": np.uint8,
        "Quality_Flag": np.uint32,
    }

    for var_name, dtype in type_map.items():
        if var_name in dataset and dataset[var_name].dtype != dtype:
            dataset[var_name] = dataset[var_name].astype(dtype)

    return dataset
