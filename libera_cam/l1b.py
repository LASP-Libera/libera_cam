"""L1b processing code libera WFOV camera"""
# Standard
import argparse
from datetime import datetime
import logging
import os
import pathlib
import time
# installed
import xarray as xr
# local
from libera_utils.io.manifest import Manifest
from libera_utils.io.filenaming import DataLevel, ManifestType
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def algorithm(parsed_cli_args: argparse.Namespace) -> str:
    """

    Parameters
    ----------
    parsed_cli_args: argparse.Namespace
        command line argument of the incoming manifest file


    Returns
    -------
    output_manifest: str
        the path of the output manifest as a string

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
        data_product_file = write_data_product(file['filename'], input_manifest.filename)

        output_manifest.add_file_to_manifest(data_product_file)
        time.sleep(1)

    logger.info("Writing the physical output manifest")
    output_manifest_filepath = output_manifest.write(pathlib.Path(os.getenv("PROCESSING_DROPBOX")))

    return output_manifest_filepath


def write_data_product(incoming_file: str, input_man: str) -> str:
    """
    Takes a file named in the input manifest and generates the output nectdf4 file, with tags and correct output name

    Parameters
    ----------
    incoming_file: str
        incoming data file retrieved from the input manifest file
    input_man
        the incoming manifest that houses the files, needed to add tags to the newly created netcdf4 files


    Returns
    -------

    data_product_filename: str
        the file path of the data product filename

    """

    logger.info("Opening the file ")
    incoming_data = xr.open_dataset(incoming_file, engine="h5netcdf")

    logger.info('Adding tags to the netcdf4 dataset')
    incoming_data.attrs['Incoming_Process_Date(UTC)'] = str(datetime.utcnow())
    incoming_data.attrs['Incoming_manifest_name'] = input_man

    timestamp = datetime.utcnow().strftime("%Y%m%dt%H%M%S")

    dropbox_path = os.getenv("PROCESSING_DROPBOX")
    data_product_filename = f"{dropbox_path}/libera_cam_{DataLevel['L1B']}_ThisIsARandDesc_" \
                            f"{timestamp}_vM1m2p3_r27002112233.h5"

    logger.info("Writing the new netcdf4 file to the output manifest")
    incoming_data.to_netcdf(data_product_filename, engine="h5netcdf")

    return data_product_filename
