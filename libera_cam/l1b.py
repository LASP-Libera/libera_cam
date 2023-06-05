"""L1b processing code libera WFOV camera"""
# Standard
import argparse
from datetime import datetime
import logging
import os
import pathlib
import time
from typing import Union
# Installed
import xarray as xr
import cloudpathlib
# Local
from libera_utils.io.manifest import Manifest
from libera_utils.io.filenaming import ManifestType, L1bFilename, PRINTABLE_TS_FORMAT
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def algorithm(parsed_cli_args: argparse.Namespace) -> Union[cloudpathlib.CloudPath, pathlib.Path]:
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
    output_manifest = Manifest(manifest_type=ManifestType.OUTPUT, files=[], configuration={})

    logger.info("Reading each file in the manifest")
    for file in input_manifest.files:

        try:
            incoming_file = file['filename']
            with smart_open(incoming_file):
                logger.info('Successfully opened file')
        except:
            logger.info('Unsuccessfully opened the file')

        logger.info("Writing the new netcdf4 file to the output manifest")
        data_product_file = write_data_product(file['filename'], input_manifest)

        output_manifest.add_file_to_manifest(data_product_file.path)
        time.sleep(1)

    logger.info("Writing the physical output manifest")
    output_manifest_filepath = output_manifest.write(pathlib.Path(os.getenv("PROCESSING_DROPBOX")))

    return output_manifest_filepath


def write_data_product(incoming_file: str, input_man: Manifest) -> L1bFilename:
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

    logger.info('Adding tags to the netcdf4 dataset')
    incoming_data.attrs['Incoming_Process_Date(UTC)'] = str(datetime.utcnow())
    incoming_data.attrs['Incoming_manifest_name'] = str(input_man.filename)

    timestamp = datetime.utcnow().strftime(PRINTABLE_TS_FORMAT)

    dropbox_path = os.getenv("PROCESSING_DROPBOX")
    data_product_filename = L1bFilename(
        f"{dropbox_path}/libera_l1b_cam_{timestamp}_{timestamp}_vM1m2p3_r27002112233.h5")

    logger.info("Writing the new netcdf4 file to the output manifest")
    incoming_data.to_netcdf(str(data_product_filename), engine="h5netcdf")

    return data_product_filename
