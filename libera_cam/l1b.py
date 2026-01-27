"""L1b processing code libera WFOV camera"""

# Standard
import argparse
import logging
import os
from datetime import datetime
from importlib import resources
from pathlib import Path

import dask
import xarray as xr
from cloudpathlib import AnyPath
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.smart_open import smart_copy_file, smart_open

from libera_cam.camera import convert_dn_to_radiance
from libera_cam.constants import DEFAULT_TIME_CHUNK_SIZE
from libera_cam.geolocation import (
    GeolocationKernelConfig,
    add_geolocation_to_dataset,
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

    # Step 2: Read and store ALL input data from manifest files
    logger.info("Step 2: Reading all input data from manifest files")
    l1a_data, spice_directory = read_all_input_data(input_manifest)

    # Step 3: Calculate science data variables (YOUR SCIENCE GOES HERE)
    logger.info("Step 3: Calculating science data variables")
    processed_data = process_l1a_to_l1b(l1a_data, spice_directory)

    # Steps 4: Store data with metadata and write to output folder
    logger.info("Step 4: Creating and writing data product")
    # This is where the compute happens!
    start = datetime.now()
    output_data_file_path = write_data_product(processed_data, dropbox_path)
    end = datetime.now()
    logger.info(f"Wrote data product in {(end - start).total_seconds()} seconds")

    # Step 6: Create output manifest
    logger.info("Step 5: Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    # Step 7: Add data files to output manifest
    logger.info("Step 6: Adding data files to output manifest")
    output_manifest.add_files(output_data_file_path.path)

    # Step 8: Write output manifest to output dropbox folder
    logger.info("Step 7: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    logger.info(f"Processing complete. Output manifest: {output_manifest_filepath}")

    return output_manifest_filepath


def read_all_input_data(input_manifest: Manifest) -> tuple[dict[str, xr.Dataset], Path]:
    """
    Read and store all input data from manifest files.

    This function opens and validates all input NetCDF files from the manifest and stores them in a dictionary keyed by
    filename. SPICE kernel files (.bc, .bsp) are copied to a local directory for processing.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest containing file information.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary with filenames as keys and loaded xarray datasets as values.
    Path
        Path to the directory of SPICE files copied to local filesystem.

    Raises
    ------
    Exception
        If any file cannot be opened or is invalid.

    Warnings
    --------
    Logs a warning if no data files were loaded from the manifest.
    """
    logger.info("Step 2: Reading all input data from manifest files")

    # Use Path object and ensure directory exists
    spice_directory = Path(__file__).parent / "spice_files"
    spice_directory.mkdir(exist_ok=True)

    all_data = {}
    spice_files_copied = []

    for i, file_info in enumerate(input_manifest.files):
        logger.info(f"Reading file {i + 1}/{len(input_manifest.files)}: {file_info.filename}")

        try:
            if file_info.filename.endswith((".bc", ".bsp")):
                local_file_destination = spice_directory / Path(file_info.filename).name
                smart_copy_file(file_info.filename, str(local_file_destination))
                spice_files_copied.append(local_file_destination)
                logger.info(f"Successfully copied SPICE file to: {local_file_destination}")
            else:
                with smart_open(file_info.filename) as file_handle:
                    dataset = xr.open_dataset(file_handle).load()
                    libera_filename = LiberaDataProductFilename(file_info.filename)
                    all_data[str(libera_filename.data_product_id)] = dataset
                    logger.info(f"Successfully loaded dataset with variables: {list(dataset.variables)}")
        except Exception as e:
            logger.error(f"Failed to process file {file_info.filename}: {e}", exc_info=True)
            raise

    logger.info(f"Successfully loaded {len(all_data)} datasets and {len(spice_files_copied)} SPICE files")

    if not all_data:
        logger.warning("No data files were loaded from manifest")

    return all_data, spice_directory


def process_l1a_to_l1b(all_input_data: dict[str, xr.Dataset], spice_directory: Path) -> xr.Dataset:
    """
    Process L1A data camera data and SPICE Kernels to L1B product.

    This function coordinates the full L1A to L1B processing pipeline including:
    - Loading calibration data
    - Extracting radiometer and housekeeping datasets
    - Initializing SPICE kernels for geolocation
    - Gain calibration of radiometer data
    - Downsampling calibrated radiometer data to 100Hz
    - Calculating geolocation information
    - Interpolating temperatures
    - Computing radiances
    - Packaging the final L1B product

    Parameters
    ----------
    all_input_data : dict[str, xr.Dataset]
        Dictionary of input datasets keyed by filename. Expected to contain radiometer sample data ('rad_sample') and
        nominal housekeeping data ('nom_hk').
    spice_directory : Path
        Path to directory containing SPICE kernel files (.bc, .bsp) for spacecraft positioning and attitude
        calculations.

    Returns
    -------
    xr.Dataset
        L1B product dataset, with variables defined by the L1B product definition.

    Raises
    ------
    ValueError
        If required input datasets (radiometer or housekeeping data) are not found.
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
    cam_dataset["Radiance"].attrs = {"long_name": "Radiance", "units": "W m^-2 sr^-1"}

    # Apply Geolocation (Lazy)
    geo_config = GeolocationKernelConfig(
        temp_dir_base=None,  # Use default temp location on workers
        dynamic_kernel_directory=spice_directory,
    )
    cam_dataset = add_geolocation_to_dataset(cam_dataset, geo_config, pixel_mask=cam_dataset.valid_pixel_mask)

    # Package output product (Renaming, Transposing, Typing)
    cam_dataset = package_l1b_product(cam_dataset)

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
    data_product_filename: L1bFilename
        The valid L1bFilename of the written data product
    """
    data_folder = resources.files("libera_cam.data")
    product_def_path = data_folder / "L1B_CAM_product_definition.yml"

    output_filename = write_libera_data_product(
        data_product_definition=product_def_path,
        data=processed_data,
        output_path=output_path,
        time_variable="camera_time",
    )

    return output_filename
