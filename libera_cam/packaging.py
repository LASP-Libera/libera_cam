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
        "n_packets_read",
        "n_images_stitched",
        "n_images_discarded_sop",
        "n_images_discarded_gap",
        "n_unexpected_eop",
        "n_images_decoded",
    ):
        dataset.attrs.pop(attr_name, None)

    # 1. Rename variables/dims to match Product Definition.
    # We map internal names (e.g. 'image_data') to public names (e.g. 'Pixel_Counts').
    dataset = dataset.rename(
        {
            "azimuth_angle": "Azimuth",
            "rad_obs_id": "Radiometer_Operational_Mode",
            "cam_obs_id": "Camera_Operational_Mode",
            "image_data": "Pixel_Counts",
            "integration_mask": "Integration_Time_Flag",
            "good_image_flag": "Quality_Flag",
            "camera_time": "CAMERA_TIME",
            "x": "CAMERA_PIXEL_COUNT_X",
            "y": "CAMERA_PIXEL_COUNT_Y",
        }
    )

    # 2. Reorder dimensions to match product definition: (Time, X, Y)
    # This effectively transposes the image arrays if they were (Time, Y, X).
    dataset = dataset.transpose("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")

    # 3. Create Placeholders for unused fields (Lazy)
    # Using da.zeros_like to match the chunking of Radiance
    # Note: Radiance is now (Time, X, Y), so placeholder will match this shape.
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
    # Using explicit casting to float32/uint types
    type_map = {
        "Azimuth": np.float32,
        "Radiometer_Operational_Mode": np.uint16,
        "Camera_Operational_Mode": np.uint16,
        "Pixel_Counts": np.uint16,
        "Integration_Time": np.uint8,
        "Quality_Flag": np.uint32,
        # Geolocation fields
        "Latitude": np.float32,
        "Longitude": np.float32,
        "Altitude": np.float32,
    }

    for var_name, dtype in type_map.items():
        if var_name in dataset:
            if dataset[var_name].dtype != dtype:
                dataset[var_name] = dataset[var_name].astype(dtype)

    # Normalize geolocation long_name metadata to match the product definition.
    if "Latitude" in dataset:
        dataset["Latitude"].attrs["long_name"] = "Geodetic latitude. Coordinate Reference System WGS84"
    if "Longitude" in dataset:
        dataset["Longitude"].attrs["long_name"] = "Longitude. Coordinate Reference System WGS84"
    if "Altitude" in dataset:
        dataset["Altitude"].attrs["long_name"] = "Height above the WGS84 ellipsoid. EPSG:4979"

    return dataset
