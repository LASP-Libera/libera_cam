"""L1b processing code libera WFOV camera"""

# Standard
import argparse
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from importlib import resources
from pathlib import Path

import dask
import xarray as xr
from cloudpathlib import AnyPath, S3Path
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest
from libera_utils.io.netcdf import write_libera_data_product

from libera_cam.camera import convert_dn_to_radiance
from libera_cam.constants import DEFAULT_TIME_CHUNK_SIZE
from libera_cam.geolocation import (
    GeolocationKernelConfig,
    add_geolocation_to_dataset,
    add_placeholder_geolocation_to_dataset,
)
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data
from libera_cam.packaging import package_l1b_product

logger = logging.getLogger(__name__)


def algorithm(parsed_cli_args: argparse.Namespace) -> AnyPath:
    """

    Parameters
    ----------
    parsed_cli_args: argparse.Namespace
        Command line argument of the incoming manifest file

    Returns
    -------
    output_manifest: Cloudpath or Path
        The path of the output manifest as a string
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

    # Determine processing mode: presence of 'ground_data' key in the manifest
    # configuration signals that SPICE kernels are unavailable and placeholder
    # geolocation should be used instead of computing from spacecraft attitude.
    no_geo_mode = "no_geo" in input_manifest.configuration
    if no_geo_mode:
        logger.info("No geolocation mode detected: placeholder geolocation will be used.")

    # Step 2: Read and store ALL input data from manifest files
    logger.info("Step 2: Reading all input data from manifest files")
    l1a_data, dynamic_kernel_sources = read_all_input_data(input_manifest, no_geo_mode=no_geo_mode)

    # Step 3: Calculate science data variables (YOUR SCIENCE GOES HERE)
    logger.info("Step 3: Calculating science data variables")
    processed_data = process_l1a_to_l1b(l1a_data, dynamic_kernel_sources, no_geo_mode=no_geo_mode)

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


def read_all_input_data(
    input_manifest: Manifest, no_geo_mode: bool = False
) -> tuple[dict[str, xr.Dataset], list[str] | None]:
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
    list[str] or None
        Manifest paths for dynamic SPICE kernels, in manifest order, or None when no_geo_mode is True and SPICE kernels
        are not required.

    Raises
    ------
    Exception
        If any file cannot be opened or is invalid.

    Warnings
    --------
    Logs a warning if no data files were loaded from the manifest.
    """
    logger.info("Step 2: Reading all input data from manifest files")

    all_data: dict[str, xr.Dataset] = {}
    dynamic_kernel_sources: list[str] | None = [] if not no_geo_mode else None

    for i, file_info in enumerate(input_manifest.files):
        logger.info(f"Reading file {i + 1}/{len(input_manifest.files)}: {file_info.filename}")

        try:
            if file_info.filename.endswith((".bc", ".bsp")):
                if no_geo_mode:
                    logger.info(f"No geolocation mode: skipping SPICE file {file_info.filename}")
                    continue
                dynamic_kernel_sources.append(file_info.filename)
                logger.info("Recorded SPICE kernel for KernelManager: %s", file_info.filename)
            else:
                dataset = xr.open_dataset(file_info.filename)
                libera_filename = LiberaDataProductFilename(file_info.filename)
                if libera_filename.data_product_id is not DataProductIdentifier.l1a_icie_wfov_sci_decoded:
                    raise ValueError(
                        f"Unexpected data product ID {libera_filename.data_product_id} in file {file_info.filename}."
                        f"Expected L1A WFOV SCI DECODED data."
                    )
                all_data[str(DataProductIdentifier.l1a_icie_wfov_sci_decoded)] = dataset
                logger.info(f"Successfully opened dataset with variables: {list(dataset.variables)}")
        except Exception as e:
            logger.error(f"Failed to process file {file_info.filename}: {e}", exc_info=True)
            raise

    logger.info(
        "Successfully opened %d datasets and %d SPICE kernels",
        len(all_data),
        0 if dynamic_kernel_sources is None else len(dynamic_kernel_sources),
    )

    if not all_data:
        logger.warning("No data files were loaded from manifest")

    return all_data, dynamic_kernel_sources


def process_l1a_to_l1b(
    all_input_data: dict[str, xr.Dataset],
    dynamic_kernel_sources: Sequence[str | Path | S3Path] | None,
    no_geo_mode: bool = False,
) -> xr.Dataset:
    """
    Process L1A camera data and SPICE Kernels to L1B product.

    This function coordinates the core L1A to L1B camera processing steps:
    - Parse the input L1A camera data into a working dataset
    - Convert DN to radiance (lazy when backed by Dask arrays)
    - Add geolocation (lazy Dask `map_blocks`), or placeholders when `no_geo_mode` is enabled

    Parameters
    ----------
    all_input_data : dict[str, xr.Dataset]
        Dictionary of input datasets keyed by filename. Expected to contain camera sample data and
        nominal housekeeping data.
    dynamic_kernel_sources : sequence of str, pathlib.Path, or cloudpathlib.S3Path, or None
        Dynamic kernel sources passed through to geolocation workers via
        :class:`~libera_cam.geolocation.GeolocationKernelConfig`.
        Each source is materialized through libera_utils `KernelFileCache` inside
        :meth:`~libera_utils.libera_spice.kernel_manager.KernelManager.load_libera_dynamic_kernels`.
        Not used when `no_geo_mode` is True.
    no_geo_mode : bool, optional
        When True, replaces SPICE-based geolocation with NaN placeholder arrays.
        Triggered by the presence of a 'no_geo' key in the input manifest
        configuration. Defaults to False (production SPICE path).

    Returns
    -------
    xr.Dataset
        L1B product dataset, with variables defined by the L1B product definition.

    Raises
    ------
    ValueError
        If required input datasets (camera or housekeeping data) are not found.
    FileNotFoundError
        If the calibration data file is not found.
    """
    # Initially, we only have WFOV-SCI-DECODED data to process
    l1a_cam_data = all_input_data[DataProductIdentifier.l1a_icie_wfov_sci_decoded]

    # Output is a tuple of (images, metadata, integration time masks)
    cam_dataset = read_l1a_cam_data(l1a_cam_data)

    # Rechunk to reduce graph size and overhead for SPICE kernel loading
    # Allow override via env var for tuning
    chunk_size = int(os.getenv("LIBERA_CAM_CHUNK_SIZE", DEFAULT_TIME_CHUNK_SIZE))
    cam_dataset = cam_dataset.chunk({"camera_time": chunk_size})

    calibrated_images = convert_dn_to_radiance(cam_dataset.image_data, cam_dataset.integration_mask)
    cam_dataset["Radiance"] = (("camera_time", "y", "x"), calibrated_images.data)

    # Apply Geolocation (Lazy)
    # No geolocation mode uses NaN placeholders because spacecraft attitude kernels
    # are not available outside of production/flight-data processing.
    if no_geo_mode:
        cam_dataset = add_placeholder_geolocation_to_dataset(cam_dataset)
    else:
        geo_config = GeolocationKernelConfig(
            temp_dir_base=None,
            dynamic_kernel_sources=dynamic_kernel_sources,
        )
        cam_dataset = add_geolocation_to_dataset(cam_dataset, geo_config, pixel_mask=cam_dataset.valid_pixel_mask)

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
    data_folder = resources.files("libera_cam.data")
    product_def_path = data_folder / "L1B_CAM_product_definition.yml"

    output_files = write_libera_data_product(
        data_product_definition=product_def_path,
        data=processed_data,
        output_path=output_path,
        time_variable="CAMERA_TIME",
    )

    return output_files
