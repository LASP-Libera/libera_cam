"""L1b processing code libera WFOV camera"""

# Standard
import argparse
import logging
import os
from datetime import datetime
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import AnyPath
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.smart_open import smart_copy_file, smart_open
from libera_utils.libera_spice.kernel_manager import KernelManager

from libera_cam.camera import convert_dn_to_radiance
from libera_cam.geolocation import calculate_all_pixel_lat_lon_altitude
from libera_cam.image_parsing.read_l1a_cam_data import read_l1a_cam_data

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
    start = datetime.now()
    l1a_data, spice_directory = read_all_input_data(input_manifest)
    end = datetime.now()
    print(f"Read all input data in {(end - start).total_seconds()} seconds")

    # Step 3: Calculate science data variables (YOUR SCIENCE GOES HERE)
    logger.info("Step 3: Calculating science data variables")
    processed_data = process_l1a_to_l1b(l1a_data, spice_directory)

    # Steps 4: Store data with metadata and write to output folder
    logger.info("Step 4: Creating and writing data product")
    start = datetime.now()
    output_data_file_path = write_data_product(processed_data, dropbox_path)
    end = datetime.now()
    print(f"Wrote data product in {(end - start).total_seconds()} seconds")

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


def process_l1a_to_l1b(all_input_data: dict[str, xr.Dataset], spice_directory: Path) -> dict[str, np.ndarray]:
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
    dict[str, np.ndarray]
        L1B product data dictionary, with variables defined by the L1B product definition.

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
    start = datetime.now()
    cam_dataset = read_l1a_cam_data(l1a_cam_data)
    end = datetime.now()
    print(f"Read L1A CAM data in {(end - start).total_seconds()} seconds")

    start = datetime.now()
    calibrated_images = convert_dn_to_radiance(cam_dataset.image_data, cam_dataset.integration_mask)
    end = datetime.now()
    print(f"Calibrated {len(cam_dataset.image_data)} images in {(end - start).total_seconds()} seconds")
    cam_dataset["calibrated_images"] = (("time", "y", "x"), calibrated_images)

    km = KernelManager()
    km.load_libera_dynamic_kernels(str(spice_directory), needs_naif_kernels=True, needs_static_kernels=True)
    # Calculate geolocation
    start = datetime.now()
    lat_lon_alt = calculate_all_pixel_lat_lon_altitude(km, cam_dataset.time.data)
    end = datetime.now()
    print(f"Calculated lat/lon/alt for {len(cam_dataset.time.data)} images in {(end - start).total_seconds()} seconds")
    km.unload_all()

    # Package output product
    start = datetime.now()
    l1b_product = _package_l1b_cam_product(cam_dataset, lat_lon_alt)
    end = datetime.now()
    print(f"Packaged L1B product in {(end - start).total_seconds()} seconds")

    return l1b_product


def _package_l1b_cam_product(cam_dataset: xr.DataArray, lat_lon_alt_data: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Package the L1B CAM product data into a dictionary.

    Parameters
    ----------
    cam_dataset: xr.Dataset
        The processed CAM dataset containing all necessary variables.
    lat_lon_alt_data: pd.DataFrame
        DataFrame containing latitude, longitude, and altitude data for good images.
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the packaged L1B CAM product data.
    """
    pixel_placeholder = np.zeros_like(cam_dataset.calibrated_images.data)

    l1b_product = {
        "camera_time": cam_dataset.time.data,
        "Latitude": lat_lon_alt_data["latitude"].astype(np.float32),
        "Terrain_Corrected_Latitude": pixel_placeholder.astype(np.float32),
        "Longitude": lat_lon_alt_data["longitude"].astype(np.float32),
        "Terrain_Corrected_Longitude": pixel_placeholder.astype(np.float32),
        "Altitude": lat_lon_alt_data["altitude"].astype(np.float32),
        "Terrain_Corrected_Altitude": pixel_placeholder.astype(np.float32),
        "Solar_Zenith_Surface": pixel_placeholder.astype(np.float32),
        "Relative_Azimuth_Surface": pixel_placeholder.astype(np.float32),
        "Viewing_Zenith_Surface": pixel_placeholder.astype(np.float32),
        "Azimuth": cam_dataset.azimuth_angle.data.astype(np.float32),
        "Radiometer_Operational_Mode": cam_dataset.rad_obs_id.data.astype(np.uint8),
        "Camera_Operational_Mode": cam_dataset.cam_obs_id.data.astype(np.uint8),
        "Pixel_Counts": cam_dataset.image_data.data.astype(np.uint16),
        "Radiance": cam_dataset.calibrated_images.data.astype(np.float32),
        "Camera_Mask": pixel_placeholder.astype(np.uint8),
        "Integration_Time": cam_dataset.integration_mask.data.astype(np.uint8),
        # TODO[LIBSDC-682] Use libera_utils quality flags
        "Quality_Flag": cam_dataset.good_image_flag.data.astype(np.uint32),
    }
    return l1b_product


def write_data_product(processed_data, output_path) -> LiberaDataProductFilename:
    """
    Takes a file named in the input manifest and generates the output nectdf4 file, with tags and correct output name
    Parameters
    ----------
    incoming_file: str
        Incoming data file retrieved from the input manifest file
    input_man: Manifest
        The input manifest that lists the input files, stored as metadata in the output netcdf4 files

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
