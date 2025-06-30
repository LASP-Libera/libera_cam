"""L1b processing code libera WFOV camera"""

# Standard
import argparse
import logging
import os
import pathlib
import time
from datetime import UTC, datetime
from importlib import metadata

import cloudpathlib

# Installed
import xarray as xr
from libera_utils.io.filenaming import LiberaDataProductFilename

# Local
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def algorithm(parsed_cli_args: argparse.Namespace) -> cloudpathlib.CloudPath | pathlib.Path:
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
    logger.info("Reading the input manifest file")
    input_manifest = Manifest.from_file(parsed_cli_args.manifest)

    logger.info("Creating output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    logger.info("Reading each file in the manifest")
    for file in input_manifest.files:
        try:
            incoming_file = file.filename
            with smart_open(incoming_file):
                logger.info("Successfully opened file")
        except Exception as excep:
            logger.info("Unsuccessfully opened the file")
            raise excep

        logger.info("Writing the new netcdf4 file to the output manifest")
        data_product_file = write_data_product(file.filename, input_manifest)

        output_manifest.add_files(data_product_file.path)
        time.sleep(1)

    logger.info("Writing the physical output manifest")
    output_manifest_filepath = output_manifest.write(pathlib.Path(os.getenv("PROCESSING_DROPBOX")))

    return output_manifest_filepath


def write_data_product(incoming_file: str, input_man: Manifest) -> LiberaDataProductFilename:
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
    logger.info("Opening the file ")
    incoming_data = xr.open_dataset(incoming_file, engine="h5netcdf")

    logger.info("Adding tags to the netcdf4 dataset")
    incoming_data.attrs["Incoming_Process_Date(UTC)"] = str(datetime.now(UTC))
    incoming_data.attrs["Incoming_manifest_name"] = str(input_man.filename)

    dropbox_path = os.getenv("PROCESSING_DROPBOX")

    current_time = datetime.now(UTC)

    product_version = metadata.version("libera_cam").replace(".", "-")
    data_product_filename = LiberaDataProductFilename.from_filename_parts(
        data_level="L1B",
        product_name="CAM",
        version=f"V{product_version}",
        utc_start=datetime(2027, 1, 1, 0, 0, 0),
        utc_end=datetime(2027, 1, 1, 1, 59, 59),
        revision=current_time,
        extension="h5",
        basepath=dropbox_path,
    )

    logger.info("Writing the new netcdf4 file to the output manifest")
    incoming_data.to_netcdf(str(data_product_filename), engine="h5netcdf")

    return data_product_filename
