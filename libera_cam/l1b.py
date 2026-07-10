"""L1b processing code libera WFOV camera"""

# Standard
import argparse
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, S3Path
from libera_utils import Manifest, smart_open
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.netcdf import write_libera_data_product

from libera_cam.camera import convert_dn_to_radiance
from libera_cam.config import product_config_path
from libera_cam.constants import DEFAULT_TIME_CHUNK_SIZE
from libera_cam.geolocation import (
    GeolocationKernelConfig,
    add_geolocation_to_dataset,
    add_jpss_only_geolocation_to_dataset,
    add_placeholder_geolocation_to_dataset,
    add_placeholder_spacecraft_geometry_to_dataset,
    add_spacecraft_geometry_to_dataset,
)
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data
from libera_cam.packaging import package_l1b_product
from libera_cam.version import version as libera_cam_version

logger = logging.getLogger(__name__)

# Required dynamic SPICE inputs keyed by Libera data product id (see libera_utils.constants).
_REQUIRED_SPICE_JPSS_ONLY: tuple[DataProductIdentifier, ...] = (
    DataProductIdentifier.spice_jpss_spk,
    DataProductIdentifier.spice_jpss_ck,
)
# Furnish order: azimuth CK before JPSS CK. ELSCAN-CK is not used by WFOV camera geolocation.
_REQUIRED_SPICE_PRODUCTION: tuple[DataProductIdentifier, ...] = (
    DataProductIdentifier.spice_az_ck,
    DataProductIdentifier.spice_jpss_spk,
    DataProductIdentifier.spice_jpss_ck,
)


def _require_spice_inputs(
    spice_files: dict[DataProductIdentifier, str],
    required: tuple[DataProductIdentifier, ...],
) -> None:
    missing = [product_id for product_id in required if product_id not in spice_files]
    if missing:
        labels = ", ".join(str(product_id) for product_id in missing)
        raise ValueError(f"Input manifest missing required SPICE data products: {labels}")


def _manifest_geo_flags(input_manifest: Manifest) -> tuple[bool, bool]:
    """Return ``(use_geo, jpss_only)`` from manifest configuration."""
    cfg = input_manifest.configuration
    use_geo = bool(cfg.get("use_geo", True))
    jpss_only_mode = bool(cfg.get("jpss_only"))
    return use_geo, jpss_only_mode


def algorithm(parsed_cli_args: argparse.Namespace) -> AnyPath:
    """
    Run the L1B camera processing pipeline from an input manifest.

    Parameters
    ----------
    parsed_cli_args: argparse.Namespace
        Command line argument of the incoming manifest file

    Returns
    -------
    output_manifest: Cloudpath or Path
        The path of the output manifest as a string

    Notes
    -----
    Manifest ``configuration.use_geo`` controls geolocation behavior. When
    ``use_geo`` is false, SPICE kernel files are skipped during input read and
    placeholder lat/lon/alt values are written. Omitting the key defaults to
    true (production SPICE geolocation). ``configuration.jpss_only`` selects
    JPSS-only SPICE geolocation (per-pixel vectors with ``LIBERA_BASE`` reference
    frame, Azimuth 0°) and cannot be combined with ``use_geo: false``.
    """
    # Enforce synchronous execution for SPICE safety and IO stability
    # 'threads' causes race conditions in CSPICE (not thread-safe).
    # 'processes' causes Pickling/IO errors with smart_open/h5netcdf.
    dask.config.set(scheduler="synchronous")

    # Set the output location to write to in the output dropbox
    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    logger.info("Reading the input manifest file")
    # Step 1: Read and use the Input Manifest
    logger.info("Step 1: Reading the input manifest file")
    input_manifest = Manifest.from_file(parsed_cli_args.manifest)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")
    use_geo, jpss_only_mode = _manifest_geo_flags(input_manifest)
    if not use_geo and jpss_only_mode:
        raise ValueError("use_geo: false and jpss_only cannot both be enabled")
    if not use_geo:
        logger.info("use_geo is false: placeholder geolocation will be used.")
    if jpss_only_mode:
        logger.info("jpss_only mode detected: LIBERA_BASE per-pixel geolocation will be used.")

    # Step 2: Read and store ALL input data from manifest files
    logger.info("Step 2: Reading all input data from manifest files")
    l1a_data, dynamic_kernel_sources = read_all_input_data(input_manifest)

    # Step 3: Calculate science data variables (YOUR SCIENCE GOES HERE)
    logger.info("Step 3: Calculating science data variables")
    processed_data = process_l1a_to_l1b(
        l1a_data,
        dynamic_kernel_sources,
        use_geo=use_geo,
        jpss_only_mode=jpss_only_mode,
    )

    # Steps 4: Store data with metadata and write to output folder
    logger.info("Step 4: Creating and writing data product")
    # This is where the compute happens!
    start = datetime.now()
    packaged_data = package_l1b_product(processed_data)
    output_files = write_data_product(packaged_data, dropbox_path)
    end = datetime.now()
    logger.info(f"Wrote data product in {(end - start).total_seconds()} seconds")

    # Step 6: Create output manifest
    logger.info("Step 5: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)
    # Propagate the full input configuration block so downstream users can
    # inspect processing mode, time ranges, and any other operator settings.
    output_manifest.configuration.update(input_manifest.configuration)

    # Step 7: Add data files to output manifest
    logger.info("Step 6: Adding data files to output manifest")
    # write_libera_data_product can return a single filename or a tuple
    if isinstance(output_files, list | tuple):
        for file in output_files:
            output_manifest.add_files(file.path)
    else:
        output_manifest.add_files(output_files.path)

    # Step 8: Write output manifest to output dropbox folder
    logger.info("Step 7: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    logger.info(f"Processing complete. Output manifest: {output_manifest_filepath}")

    return output_manifest_filepath


def read_all_input_data(input_manifest: Manifest) -> tuple[dict[str, xr.Dataset], list[str]]:
    """
    Read and store all input data from manifest files.

    This function opens and validates all input NetCDF files from the manifest and stores them in a dictionary keyed by
    filename. SPICE kernel sources (.bc, .bsp) are collected in manifest order for later use with
    :meth:`~libera_utils.libera_spice.kernel_manager.KernelManager.load_libera_dynamic_kernels`, which materializes each
    source via libera_utils `KernelFileCache` (local file, S3, or HTTP(S) as supported).

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest containing file information.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary with filenames as keys and loaded xarray datasets as values.
    list[str]
        Manifest paths for dynamic SPICE kernels, in furnish order. Empty when
        ``input_manifest.configuration.use_geo`` is false and SPICE kernels are
        not required. When ``configuration.jpss_only`` is true, only JPSS-SPK
        and JPSS-CK paths are collected.

    Raises
    ------
    Exception
        If any file cannot be opened or is invalid.
    ValueError
        If duplicate SPICE data products appear in the manifest, or required
        kernels are missing when ``use_geo`` is true.

    Warnings
    --------
    Logs a warning if no data files were loaded from the manifest.

    Notes
    -----
    When ``input_manifest.configuration.use_geo`` is false, SPICE kernel files
    (.bc, .bsp) are skipped. Omitting ``use_geo`` defaults to true.
    """
    logger.info("Step 2: Reading all input data from manifest files")

    use_geo, jpss_only_mode = _manifest_geo_flags(input_manifest)
    all_data: dict[str, xr.Dataset] = {}
    spice_files: dict[DataProductIdentifier, str] = {}

    for i, file_info in enumerate(input_manifest.files):
        logger.info(f"Reading file {i + 1}/{len(input_manifest.files)}: {file_info.filename}")

        try:
            if file_info.filename.endswith((".bc", ".bsp")):
                if not use_geo:
                    logger.warning(
                        "use_geo is false: skipping SPICE kernel %s",
                        file_info.filename,
                    )
                    continue

                product_id = LiberaDataProductFilename.from_file_path(file_info.filename).data_product_id
                if jpss_only_mode and product_id not in _REQUIRED_SPICE_JPSS_ONLY:
                    logger.warning(
                        "jpss_only mode: skipping SPICE file %s (%s)",
                        file_info.filename,
                        product_id,
                    )
                    continue

                if product_id in spice_files:
                    raise ValueError(
                        f"Duplicate SPICE data product {product_id} in manifest: "
                        f"{spice_files[product_id]} and {file_info.filename}"
                    )

                spice_files[product_id] = file_info.filename
                logger.info(
                    "Recorded SPICE kernel %s (%s)",
                    file_info.filename,
                    product_id,
                )
            else:
                with smart_open(file_info.filename) as file_handle:
                    LiberaDataProductFilename.from_file_path(file_info.filename)  # Ensure file is Libera Data Product
                    dataset = xr.open_dataset(file_handle, decode_times=True).load()
                    all_data[file_info.filename] = dataset
                    logger.info(f"Successfully loaded dataset: {file_handle}")
        except Exception as e:
            logger.error(f"Failed to process file {file_info.filename}: {e}", exc_info=True)
            raise

    dynamic_kernel_sources: list[str] = []
    if use_geo:
        required_spice = _REQUIRED_SPICE_JPSS_ONLY if jpss_only_mode else _REQUIRED_SPICE_PRODUCTION
        _require_spice_inputs(spice_files, required_spice)
        dynamic_kernel_sources = [spice_files[product_id] for product_id in required_spice]

    logger.info(
        "Successfully opened %d datasets and %d SPICE kernel paths",
        len(all_data),
        len(dynamic_kernel_sources),
    )

    if not all_data:
        logger.warning("No data files were loaded from manifest")

    return all_data, dynamic_kernel_sources


def _extract_camera_dataset(all_input_data: dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Extract WFOV SCI DECODED camera dataset from input data.

    Searches through the input data dictionary to identify the L1A WFOV science
    dataset based on filename product id, matching the libera_rad pattern of
    keying ``read_all_input_data`` results by manifest filename.
    """
    for file_name, dataset in all_input_data.items():
        libera_filename = LiberaDataProductFilename.from_file_path(file_name)
        if libera_filename.data_product_id == DataProductIdentifier.l1a_icie_wfov_sci_decoded.value:
            return dataset

    raise ValueError("No WFOV SCI DECODED data found in input files")


def _apply_azimuth_fill(ds: xr.Dataset, fill_value: float) -> xr.Dataset:
    """Replace ``azimuth_angle`` with a constant per-frame fill value."""
    if "azimuth_angle" not in ds:
        return ds
    n_times = ds.sizes["camera_time"]
    ds["azimuth_angle"] = (
        ("camera_time",),
        np.full(n_times, fill_value, dtype=np.float32),
    )
    return ds


def process_l1a_to_l1b(
    all_input_data: dict[str, xr.Dataset],
    dynamic_kernel_sources: Sequence[str | Path | S3Path],
    use_geo: bool = True,
    jpss_only_mode: bool = False,
) -> xr.Dataset:
    """
    Process L1A camera data and SPICE Kernels to L1B product.

    This function coordinates the core L1A to L1B camera processing steps:
    - Parse the input L1A camera data into a working dataset
    - Convert DN to radiance (lazy when backed by Dask arrays)
    - Add geolocation (lazy Dask ``map_blocks``), JPSS-only LIBERA_BASE geolocation,
      or placeholders when ``use_geo`` is false
    - Add the spacecraft-level geometry (sub-point, satellite radius, Earth-Sun distance),
      or placeholders when ``use_geo`` is false

    Parameters
    ----------
    all_input_data : dict[str, xr.Dataset]
        Dictionary of input datasets keyed by filename. Expected to contain camera sample data and
        nominal housekeeping data.
    dynamic_kernel_sources : sequence of str, pathlib.Path, or cloudpathlib.S3Path
        Dynamic kernel sources passed through to geolocation workers via
        :class:`~libera_cam.geolocation.GeolocationKernelConfig`.
        Each source is materialized through libera_utils `KernelFileCache` inside
        :meth:`~libera_utils.libera_spice.kernel_manager.KernelManager.load_libera_dynamic_kernels`.
        May be empty when ``use_geo`` is false.
    use_geo : bool, optional
        When True (default), runs SPICE geolocation. When False, uses placeholder
        lat/lon/alt for ground-calibration processing. Set via manifest
        ``configuration.use_geo``; omitting the key is equivalent to True.
    jpss_only_mode : bool, optional
        When True, uses per-pixel geolocation with ``LIBERA_BASE`` (zero-azimuth
        approximation) and sets Azimuth to 0°. Requires ``use_geo`` True and JPSS-only
        SPICE kernels in the manifest.

    Returns
    -------
    xr.Dataset
        L1B product dataset, with variables defined by the L1B product definition.

    Raises
    ------
    ValueError
        If required input datasets (camera or housekeeping data) are not found,
        or SPICE kernel sources are missing when ``use_geo`` is True.
    FileNotFoundError
        If the calibration data file is not found.
    """
    l1a_cam_data = _extract_camera_dataset(all_input_data)

    # Output is a tuple of (images, metadata, integration time masks)
    cam_dataset = read_l1a_cam_data(l1a_cam_data)

    # Rechunk to reduce graph size and overhead for SPICE kernel loading
    # Allow override via env var for tuning
    chunk_size = int(os.getenv("LIBERA_CAM_CHUNK_SIZE", DEFAULT_TIME_CHUNK_SIZE))
    cam_dataset = cam_dataset.chunk({"camera_time": chunk_size})

    calibrated_images = convert_dn_to_radiance(cam_dataset.image_data, cam_dataset.integration_mask)
    cam_dataset["Radiance"] = (("camera_time", "y", "x"), calibrated_images.data)

    # Apply Geolocation (Lazy) and the spacecraft-level geometry (eager, one value per frame).
    # The spacecraft geometry call is the same in both SPICE modes: its fields need only the
    # spacecraft ephemeris, never the camera's instrument frame.
    if not use_geo:
        cam_dataset = add_placeholder_geolocation_to_dataset(cam_dataset)
        cam_dataset = add_placeholder_spacecraft_geometry_to_dataset(cam_dataset)
        cam_dataset = _apply_azimuth_fill(cam_dataset, fill_value=-999.0)
    elif jpss_only_mode:
        if not dynamic_kernel_sources:
            raise ValueError("SPICE kernel sources are required for geolocation when jpss_only_mode is True")
        geo_config = GeolocationKernelConfig(
            temp_dir_base=None,
            dynamic_kernel_sources=dynamic_kernel_sources,
        )
        cam_dataset = add_jpss_only_geolocation_to_dataset(
            cam_dataset, geo_config, pixel_mask=cam_dataset.valid_pixel_mask
        )
        cam_dataset = add_spacecraft_geometry_to_dataset(cam_dataset, geo_config)
        cam_dataset = _apply_azimuth_fill(cam_dataset, fill_value=0.0)
    else:
        if not dynamic_kernel_sources:
            raise ValueError("SPICE kernel sources are required for geolocation when use_geo is True")
        geo_config = GeolocationKernelConfig(
            temp_dir_base=None,
            dynamic_kernel_sources=dynamic_kernel_sources,
        )
        cam_dataset = add_geolocation_to_dataset(cam_dataset, geo_config, pixel_mask=cam_dataset.valid_pixel_mask)
        cam_dataset = add_spacecraft_geometry_to_dataset(cam_dataset, geo_config)

    return cam_dataset


def write_data_product(processed_data: xr.Dataset, output_path: str) -> LiberaDataProductFilename:
    """
    Takes a file named in the input manifest and generates the output nectdf4 file, with tags and correct output name
    Parameters
    ----------
    processed_data: xr.Dataset
        The dataset to write
    output_path: str
        The path to write the output file to

    Returns
    -------
    data_product_filenames: LiberaDataProductFilename
        The valid filename of the written data product(s)
    """
    if not product_config_path.exists():
        raise FileNotFoundError(f"Product definition file not found: {product_config_path}")

    processed_data.attrs["algorithm_version"] = libera_cam_version()

    output_files = write_libera_data_product(
        data_product_definition=product_config_path,
        data=processed_data,
        output_path=output_path,
        time_variable="CAMERA_TIME",
    )

    return output_files
