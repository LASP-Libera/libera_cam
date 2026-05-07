"""
Geolocation Calculations for the Libera Wide-field of View (WFOV) Camera.

This module provides a clean interface for managing SPICE kernels and performing
geolocation calculations for the Libera camera.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import S3Path
from curryer import spicetime
from curryer import spicierpy as sp
from curryer.compute import spatial
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.constants import GROUND_CAL_PIXEL_MAPPING, PIXEL_COUNT_X, PIXEL_COUNT_Y

logger = logging.getLogger(__name__)


@dataclass
class GeolocationKernelConfig:
    """
    Configuration for initializing a KernelManager on a worker node.

    This dataclass is pickleable and can be passed to Dask workers.

    Parameters
    ----------
    dynamic_kernel_sources : pathlib.Path, str, or sequence, optional
        Dynamic SPICE kernels to load for geolocation. May be either:
        - a directory containing kernel files (non-recursive),
        - a single kernel file path, or
        - an explicit sequence of sources (e.g. manifest-ordered `.bc` / `.bsp` paths, including S3).

        Each source is materialized through libera_utils `KernelFileCache` inside
        :meth:`libera_utils.libera_spice.kernel_manager.KernelManager.load_libera_dynamic_kernels`.
    """

    temp_dir_base: str | Path | None = None
    download_naif_url: str = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
    use_test_naif_url: bool = False
    use_high_precision_earth: bool = True
    cache_timeout_days: int = 7
    dynamic_kernel_sources: str | Path | Sequence[str | Path | S3Path] | None = None


def prefetch_kernels(config: GeolocationKernelConfig) -> None:
    """
    Downloads/Generates kernels on the client to populate the cache safely.

    This function is designed to run serially on the client process before
    scattering work to Dask workers. This prevents race conditions where
    multiple workers attempt to download/write the same kernel files simultaneously,
    which can lead to file corruption (as KernelFileCache lacks locking) and
    excessive load on the NAIF servers.
    """
    logger.info("Pre-fetching SPICE kernels to populate local cache...")
    km = KernelManager(
        temp_dir_base=config.temp_dir_base,
        download_naif_url=config.download_naif_url,
        use_test_naif_url=config.use_test_naif_url,
        use_high_precision_earth=config.use_high_precision_earth,
        cache_timeout_days=config.cache_timeout_days,
    )
    try:
        # Load NAIF generic kernels (downloads if missing)
        km.load_naif_kernels()
        # Load Static kernels (generates if missing)
        km.load_static_kernels()
        logger.info("Kernel cache populated successfully.")
    except Exception as e:
        logger.warning(f"Kernel pre-fetch failed: {e}. Workers will attempt download independently.")
    finally:
        # Crucial: Unload kernels to leave the client process clean.
        # We don't want to pollute the global SPICE pool if the client
        # later does other SPICE operations.
        km.unload_all()


def calculate_all_pixel_lat_lon_altitude(
    kernel_manager: KernelManager,
    image_times: list[xr.DataArray] | pd.DatetimeIndex,
    pointing_vectors: np.ndarray | None = None,
    pixel_mask: np.ndarray | None = None,
    is_dynamic_mask: bool | None = None,
) -> dict[str, np.ndarray]:
    """
    Calculate latitude, longitude, and altitude for all pixels at given image times.

    This function internally handles adapting time formats and reshaping outputs.
    For optimized performance in distributed environments, consider using `add_geolocation_to_dataset`.

    Parameters
    ----------
    kernel_manager: KernelManager
        An instance of KernelManager with kernels already loaded.
    image_times: pd.DatetimeIndex or list[xr.DataArray]
        A set of image times for which to calculate geolocation data.
        If a list of `xr.DataArray` is provided, each `DataArray` is expected to contain a single
        `np.datetime64` value, and these will be converted to a `pd.DatetimeIndex`.
    pointing_vectors: np.ndarray, optional
        Pre-loaded 2D array of pointing vectors with shape `(N_pixels, 3)`.
        If None, they are loaded from the package data `wfov_pixel_vectors.npy`
        and reshaped to `(PIXEL_COUNT_X * PIXEL_COUNT_Y, 3)`.
    pixel_mask: np.ndarray, optional
        Boolean mask where True indicates pixels to calculate.
        Can be:
        - 2D (Static): `(PIXEL_COUNT_X, PIXEL_COUNT_Y)` or flattened to `(PIXEL_COUNT_X * PIXEL_COUNT_Y,)`.
        - 3D (Dynamic): `(N_times, PIXEL_COUNT_X, PIXEL_COUNT_Y)` or flattened to
          `(N_times, PIXEL_COUNT_X * PIXEL_COUNT_Y,)`.
        Pixels corresponding to `False` in the mask will have their geolocation filled with NaN.
        If `None`, geolocation is calculated for all pixels.
    is_dynamic_mask: bool, optional
        If True, treats `pixel_mask` as a per-timestamp mask.
        If False, treats `pixel_mask` as a static mask applied to all timestamps.
        If None (default), attempts to detect based on dimensions.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the following keys, each mapped to a NumPy array
        with shape `(N_times, PIXEL_COUNT_X, PIXEL_COUNT_Y)`:
        - "latitude" (np.float64): Latitude values in degrees north.
        - "longitude" (np.float64): Longitude values in degrees east.
        - "altitude" (np.float64): Altitude values in kilometers.
    """
    kernel_manager.ensure_known_kernels_are_furnished()

    # Could use more error handling here to support more options of inputs for time range
    if isinstance(image_times, list):
        image_strings = [str(time.values) for time in image_times]
        image_times = pd.to_datetime(image_strings)
    elif isinstance(image_times, xr.DataArray):
        image_times = pd.to_datetime(image_times.values)

    # Ensure times are flat
    times_np = pd.DatetimeIndex(image_times).values.ravel()
    n_times = len(times_np)

    if pointing_vectors is None:
        # Pixel pointing vectors for the Libera WFOV camera
        pointing_vectors = np.load(GROUND_CAL_PIXEL_MAPPING, mmap_mode="r")
        pointing_vectors = pointing_vectors.reshape(-1, 3)

    n_pixels = pointing_vectors.shape[0]

    # Initialize full arrays with NaN
    full_lat = np.full((n_times, n_pixels), np.nan, dtype=np.float64)
    full_lon = np.full((n_times, n_pixels), np.nan, dtype=np.float64)
    full_alt = np.full((n_times, n_pixels), np.nan, dtype=np.float64)

    # Detect Mask Type if not provided
    if is_dynamic_mask is None:
        if pixel_mask is not None:
            if pixel_mask.ndim == 3 or (pixel_mask.ndim == 2 and pixel_mask.shape[0] == n_times):
                is_dynamic_mask = True
            else:
                is_dynamic_mask = False
        else:
            is_dynamic_mask = False

    # Normalize pixel_mask shape
    if pixel_mask is not None:
        if is_dynamic_mask:
            pixel_mask = pixel_mask.reshape(n_times, -1)
        else:
            pixel_mask = pixel_mask.ravel()

    # --- Path 1: Dynamic Mask (Loop over time) ---
    if is_dynamic_mask:
        logger.debug("Using dynamic (per-timestamp) geolocation masking.")

        # SPICE is not thread-safe within a single process.
        # We process frames serially within this worker task.
        # Parallelization is achieved at the chunk level by Dask.
        unified_gps_times = spicetime.adapt(image_times, "iso")
        unified_gps_times = np.asanyarray(unified_gps_times).ravel()

        for i in range(n_times):
            mask_t = pixel_mask[i]
            active_vectors = pointing_vectors[mask_t]

            if active_vectors.size > 0:
                cam_pix_lla, _, _ = spatial.compute_ellipsoid_intersection(
                    np.array([unified_gps_times[i]]),
                    sp.obj.Body("LIBERA_WFOV_CAM", frame=True),
                    custom_pointing_vectors=active_vectors,
                    give_geodetic_output=True,
                    give_lat_lon_in_degrees=True,
                )

                full_lat[i, mask_t] = cam_pix_lla["lat"].values
                full_lon[i, mask_t] = cam_pix_lla["lon"].values
                full_alt[i, mask_t] = cam_pix_lla["alt"].values

    # --- Path 2: Static Mask or No Mask (Vectorized) ---
    else:
        logger.debug("Using static (vectorized) geolocation masking.")
        if pixel_mask is not None:
            active_vectors = pointing_vectors[pixel_mask]
        else:
            active_vectors = pointing_vectors

        if active_vectors.size > 0:
            unified_gps_times = spicetime.adapt(image_times, "iso")
            unified_gps_times = np.asanyarray(unified_gps_times).ravel()

            cam_pix_lla, _, _ = spatial.compute_ellipsoid_intersection(
                unified_gps_times,
                sp.obj.Body("LIBERA_WFOV_CAM", frame=True),
                custom_pointing_vectors=active_vectors,
                give_geodetic_output=True,
                give_lat_lon_in_degrees=True,
            )

            # Reshape results to (n_times, n_active_pixels)
            computed_lat = cam_pix_lla["lat"].values.reshape(n_times, -1)
            computed_lon = cam_pix_lla["lon"].values.reshape(n_times, -1)
            computed_alt = cam_pix_lla["alt"].values.reshape(n_times, -1)

            if pixel_mask is not None:
                # Scatter back to full array
                full_lat[:, pixel_mask] = computed_lat
                full_lon[:, pixel_mask] = computed_lon
                full_alt[:, pixel_mask] = computed_alt
            else:
                full_lat = computed_lat
                full_lon = computed_lon
                full_alt = computed_alt

    return_dict = {
        "latitude": full_lat.reshape(n_times, PIXEL_COUNT_X, PIXEL_COUNT_Y),
        "longitude": full_lon.reshape(n_times, PIXEL_COUNT_X, PIXEL_COUNT_Y),
        "altitude": full_alt.reshape(n_times, PIXEL_COUNT_X, PIXEL_COUNT_Y),
    }

    return return_dict


def calculate_chunk_geolocation(
    camera_time: np.ndarray,
    config: GeolocationKernelConfig,
    pixel_mask: np.ndarray | None = None,
    is_dynamic_mask: bool | None = None,
) -> np.ndarray:
    """
    Worker function to compute geolocation for a chunk of times.

    This function instantiates a local KernelManager, loads kernels, computes
    geolocation, and then rigorously cleans up (unloads kernels).

    Parameters
    ----------
    camera_time : np.ndarray
        Array of camera times (datetime64[ns]) for this chunk.
    config : GeolocationKernelConfig
        Configuration for initializing the KernelManager.
    pixel_mask : np.ndarray, optional
        Boolean mask to skip calculation for invalid pixels.
    is_dynamic_mask : bool, optional
        Whether the mask is dynamic or static.

    Returns
    -------
    np.ndarray
        Array of shape (T, Y, X, 3) containing [Latitude, Longitude, Altitude].
    """
    # Instantiate fresh manager for this process
    km = KernelManager(
        temp_dir_base=config.temp_dir_base,
        download_naif_url=config.download_naif_url,
        use_test_naif_url=config.use_test_naif_url,
        use_high_precision_earth=config.use_high_precision_earth,
        cache_timeout_days=config.cache_timeout_days,
    )

    # MEMORY OPTIMIZATION: Load pointing vectors inside the worker using mmap.
    # This prevents redundant serialization of the 48MB array from the client to every task.
    pointing_vectors = np.load(GROUND_CAL_PIXEL_MAPPING, mmap_mode="r")
    pointing_vectors = pointing_vectors.reshape(-1, 3)

    # Use context manager for automatic cleanup of static kernels and unloading
    with km:
        # Load dynamic kernels if specified
        if config.dynamic_kernel_sources:
            km.load_libera_dynamic_kernels(config.dynamic_kernel_sources)

        # Convert numpy datetime64 array to DatetimeIndex for the calculation function
        flat_camera_time = np.asarray(camera_time).ravel()
        times = pd.DatetimeIndex(flat_camera_time)

        # Perform calculation
        # Result is a dict of arrays of shape (T, Y, X)
        result_dict = calculate_all_pixel_lat_lon_altitude(
            km, times, pointing_vectors, pixel_mask=pixel_mask, is_dynamic_mask=is_dynamic_mask
        )

        # Stack into (T, Y, X, 3)
        # Order: Latitude, Longitude, Altitude
        stacked_result = np.stack(
            [
                result_dict["latitude"],
                result_dict["longitude"],
                result_dict["altitude"],
            ],
            axis=-1,
        )

    return stacked_result


def add_geolocation_to_dataset(
    ds: xr.Dataset,
    config: GeolocationKernelConfig,
    pixel_mask: np.ndarray | da.Array | xr.DataArray | None = None,
    is_dynamic_mask: bool | None = None,
) -> xr.Dataset:
    """
    Lazily compute geolocation data and add it to the dataset.

    Uses Dask map_blocks to parallelize computation over the 'camera_time' dimension.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing 'camera_time' coordinate.
    config : GeolocationKernelConfig
        Configuration for SPICE kernel management.
    pixel_mask : Union[np.ndarray, da.Array, xr.DataArray, None], optional
        Boolean mask where False indicates pixels to skip (fill with NaN).
        Can be:
        - 2D (Static): (2048, 2048) np.ndarray or da.Array
        - 3D (Dynamic): (Time, 2048, 2048) da.Array or xr.DataArray matching ds.camera_time
        If None, computes all pixels.
    is_dynamic_mask : bool, optional
        Force treatment of pixel_mask as dynamic (Time, Y, X) or static (Y, X).
        If None, automatically detects based on dimensions.

    Returns
    -------
    xr.Dataset
        The dataset with added 'latitude', 'longitude', 'altitude' variables.
    """
    if "camera_time" not in ds.coords:
        raise ValueError("Dataset must have 'camera_time' coordinate.")

    # 1. Safe Pre-fetch (Client Side)
    # Ensures kernel cache is populated serially to avoid race conditions on workers
    prefetch_kernels(config)

    # 2. Determine Time Chunks
    # We want to align with image_data chunks if possible for efficiency
    if "image_data" in ds and isinstance(ds.image_data.data, da.Array):
        # image_data chunks: (time_chunks, y_chunks, x_chunks)
        time_chunks_tuple = ds.image_data.chunks[0]
    else:
        # Fallback: Chunk size of 1 frame (safe default for SPICE)
        n_times = ds.sizes["camera_time"]
        time_chunks_tuple = (1,) * n_times

    # 3. Wrap Time in Dask Array
    # This drives the map_blocks iteration
    times_da = da.from_array(ds.camera_time.values, chunks=(time_chunks_tuple,))

    # 4. Handle Pixel Mask (Static vs Dynamic)
    # Note: We no longer load pointing_vectors here. They are loaded in the workers.
    map_blocks_args = [config]

    # Determine new axes for map_blocks
    # Default: Input is 1D (Time), Output is 4D (Time, Y, X, 3) -> Add axes 1, 2, 3
    new_axes_indices = [1, 2, 3]

    if pixel_mask is not None:
        if isinstance(pixel_mask, xr.DataArray):
            pixel_mask_da = pixel_mask.data
        elif isinstance(pixel_mask, da.Array):
            pixel_mask_da = pixel_mask
        else:
            # Wrap numpy array in dask array, but don't chunk time dimension (it's static)
            pixel_mask_da = da.from_array(pixel_mask, chunks=pixel_mask.shape)

        # Detect mask type if not specified
        if is_dynamic_mask is None:
            is_dynamic_mask = pixel_mask_da.ndim == 3

        # Handle Alignment
        if is_dynamic_mask:
            # Ensure time chunks match exactly
            pixel_mask_da = pixel_mask_da.rechunk({0: time_chunks_tuple})
            map_blocks_args.append(pixel_mask_da)

            # If input is 1D (Time), Output is 4D (Time, Y, X, 3) -> Add axes 1, 2, 3
            # Dask aligns 1D and 3D arrays by right-broadcasting.
            # We want Time (Dim 0) to align with Time (Dim 0).
            # So we must reshape times_da to 3D: (Time, 1, 1).
            times_da = times_da[:, None, None]
            new_axes_indices = [3]  # Only adding the component axis
        else:
            # Static mask, pass as constant (arg) or compute and pass
            map_blocks_args.append(pixel_mask_da.compute())
    else:
        map_blocks_args.append(None)

    map_blocks_args.append(is_dynamic_mask)

    # 5. Map Blocks over Time
    # Input chunk: (Time_Chunk,) or (Time_Chunk, 1, 1)
    # Output chunk: (Time_Chunk, Y, X, 3)

    # Explicitly format chunks as tuple of tuples for all dimensions
    output_chunks = (time_chunks_tuple, (PIXEL_COUNT_Y,), (PIXEL_COUNT_X,), (3,))

    geo_data = da.map_blocks(
        calculate_chunk_geolocation,
        times_da,
        *map_blocks_args,
        dtype=np.float64,
        chunks=output_chunks,
        new_axis=new_axes_indices,
    )

    # 7. Assign variables to Dataset
    ds["Latitude"] = (("camera_time", "y", "x"), geo_data[..., 0].astype(np.float32))
    ds["Longitude"] = (("camera_time", "y", "x"), geo_data[..., 1].astype(np.float32))
    ds["Altitude"] = (("camera_time", "y", "x"), geo_data[..., 2].astype(np.float32))

    ds["Latitude"].attrs = {"units": "degrees_north", "long_name": "Pixel Latitude"}
    ds["Longitude"].attrs = {"units": "degrees_east", "long_name": "Pixel Longitude"}
    ds["Altitude"].attrs = {"units": "km", "long_name": "Pixel Altitude"}

    return ds


def add_placeholder_geolocation_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Add NaN-filled placeholder geolocation variables to the dataset.

    Used in ground-data mode when SPICE kernels are unavailable (e.g. during
    ground testing where spacecraft attitude kernels do not exist). Adds
    Latitude, Longitude, and Altitude variables filled with NaN to indicate
    that geolocation was not computed for this run.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing 'camera_time', 'y', and 'x' dimensions.

    Returns
    -------
    xr.Dataset
        The dataset with NaN-filled Latitude, Longitude, and Altitude variables
        matching the chunking of the existing image data.
    """
    if "image_data" in ds and isinstance(ds["image_data"].data, da.Array):
        time_chunks_tuple = ds["image_data"].chunks[0]
    else:
        n_times = ds.sizes["camera_time"]
        time_chunks_tuple = (1,) * n_times

    placeholder = da.full(
        (ds.sizes["camera_time"], PIXEL_COUNT_X, PIXEL_COUNT_Y),
        fill_value=np.nan,
        dtype=np.float32,
        chunks=(time_chunks_tuple, (PIXEL_COUNT_X,), (PIXEL_COUNT_Y,)),
    )

    ds["Latitude"] = (("camera_time", "y", "x"), placeholder)
    ds["Longitude"] = (("camera_time", "y", "x"), placeholder)
    ds["Altitude"] = (("camera_time", "y", "x"), placeholder)

    ds["Latitude"].attrs = {"units": "degrees_north", "long_name": "Pixel Latitude"}
    ds["Longitude"].attrs = {"units": "degrees_east", "long_name": "Pixel Longitude"}
    ds["Altitude"].attrs = {"units": "km", "long_name": "Pixel Altitude"}

    logger.info("Ground data mode: added NaN placeholder geolocation (Latitude, Longitude, Altitude).")

    return ds
