"""
Module for packaging the L1B CAM product to ensure conformance with the product definition.
"""

import logging

import dask.array as da
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def package_l1b_product(dataset: xr.Dataset) -> xr.Dataset:
    """
    Refines the internal processing dataset into the final L1B product structure.

    Responsibilities:
    1. Rename internal dimensions and variables to match the L1B Product Definition.
    2. Transpose dimensions to match the required order (Time, X, Y).
    3. Create and assign placeholder variables for fields not yet computed.
    4. Enforce data types (float32, uint8, etc.).

    Parameters
    ----------
    dataset : xr.Dataset
        The intermediate dataset containing science results (Radiance, Geolocation).

    Returns
    -------
    xr.Dataset
        The packaged dataset ready for NetCDF writing.
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

    # 1. Rename variables/dims to match Product Definition.
    rename_map = {
        "azimuth_angle": "Azimuth",
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
        "camera_time": "CAMERA_TIME",
        "x": "CAMERA_PIXEL_COUNT_X",
        "y": "CAMERA_PIXEL_COUNT_Y",
    }
    dataset = dataset.rename({src: dst for src, dst in rename_map.items() if src in dataset})

    # 2. Reorder dimensions to match product definition: (Time, X, Y)
    dataset = dataset.transpose("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")

    # 3. Create Placeholders for unused fields (Lazy)
    if "Radiance" not in dataset:
        raise ValueError("Dataset must contain 'Radiance' variable before packaging.")

    pixel_placeholder = da.zeros_like(dataset["Radiance"].data, dtype=np.float32)
    dims_3d = ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")

    placeholders = {
        "Terrain_Corrected_Latitude": (dims_3d, pixel_placeholder),
        "Terrain_Corrected_Longitude": (dims_3d, pixel_placeholder),
        "Terrain_Corrected_Altitude": (dims_3d, pixel_placeholder),
        "Solar_Zenith_Surface": (dims_3d, pixel_placeholder),
        "Relative_Azimuth_Surface": (dims_3d, pixel_placeholder),
        "Viewing_Zenith_Surface": (dims_3d, pixel_placeholder),
        "Camera_Mask": (dims_3d, pixel_placeholder.astype(np.uint8)),
    }

    for name, (dims, data) in placeholders.items():
        dataset[name] = (dims, data)

    # 4. Ensure Types (Cast if necessary)
    type_map = {
        "Azimuth": np.float32,
        "Radiometer_Observation_ID": np.uint16,
        "Camera_Observation_ID": np.uint16,
        "Image_Mode": np.uint8,
        "Camera_Packet_Index": np.int32,
        "Actual_Exposure_Time_1": np.float32,
        "Actual_Exposure_Time_2": np.float32,
        "Exposure_Delta": np.float32,
        "Pixel_Counts": np.uint16,
        "Quality_Flag": np.uint32,
        "Latitude": np.float32,
        "Longitude": np.float32,
        "Altitude": np.float32,
    }

    for var_name, dtype in type_map.items():
        if var_name in dataset and dataset[var_name].dtype != dtype:
            dataset[var_name] = dataset[var_name].astype(dtype)

    # Normalize geolocation long_name metadata to match the product definition.
    if "Latitude" in dataset:
        dataset["Latitude"].attrs["long_name"] = "Geodetic latitude. Coordinate Reference System WGS84"
    if "Longitude" in dataset:
        dataset["Longitude"].attrs["long_name"] = "Longitude. Coordinate Reference System WGS84"
    if "Altitude" in dataset:
        dataset["Altitude"].attrs["long_name"] = "Height above the WGS84 ellipsoid. EPSG:4979"
        dataset["Altitude"].attrs["units"] = "meters"

    return dataset
