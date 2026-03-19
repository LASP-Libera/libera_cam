"""The module for hdf5 helper functions for reading variables and values. Should be moved to libera_utils ant tested"""

import h5py
import numpy as np
from cloudpathlib import AnyPath


def load_hdf5_variable(
    variable_name: str, file_path: str or AnyPath = None, hdf_object: h5py.Dataset = None
) -> np.ndarray:
    """A function to read in the dark correction data from an HDF5 file.
    This will change as we store our data in NetCDF files

    Parameters
    ----------
    variable_name: str
        The name of the variable in the HDF5 file
    file_path: str or AnyPath
        The path to the HDF5 file
    hdf_object: h5py.Dataset, Optional
        The Dataset to extract a variable from

    Returns
    -------
    dark_offset: np.ndarray
        The dark offset data
    """
    if file_path is not None:
        return load_hdf5_variable_from_file(variable_name, file_path)

    if hdf_object is not None:
        return load_hdf5_variable_from_object(variable_name, hdf_object)

    return ValueError("You must specify either a file_path or an hdf_object to read")


def load_hdf5_variable_from_file(variable_name: str, file_path: str or AnyPath) -> np.ndarray:
    """A function to read in the dark correction data from an HDF5 file.
    This will change as we store our data in NetCDF files

    Parameters
    ----------
    variable_name: str
        The name of the variable in the HDF5 file
    file_path: str or AnyPath
        The path to the HDF5 file

    Returns
    -------
    dark_offset: np.ndarray
        The dark offset data
    """

    data_object = h5py.File(file_path, "r")
    dark_offset = load_hdf5_variable_from_object(variable_name, data_object)
    data_object.close()

    return dark_offset


def load_hdf5_variable_from_object(variable_name: str, hdf5_dataset: h5py.Dataset) -> np.ndarray:
    """A function to read in the dark correction data from an HDF5 file.
    This will change as we store our data in NetCDF files

    Parameters
    ----------
    variable_name: str
        The name of the variable in the HDF5 file
    hdf5_dataset: h5py.Dataset
        The Dataset to extract a variable from

    Returns
    -------
    dark_offset: np.ndarray
        The dark offset data
    """
    return hdf5_dataset[variable_name][:]


def load_hdf5_single_value(variable_name: str, file_path: str or AnyPath, hdf_object: h5py.Dataset) -> np.ndarray:
    """A function to read in the dark correction data from an HDF5 file.
    This will change as we store our data in NetCDF files

    Parameters
    ----------
    variable_name: str
        The name of the variable in the HDF5 file
    file_path: str or AnyPath, Optional
        The path to the HDF5 file
    hdf_object: h5py.Dataset, Optional
        The h5py Dataset to extract a value from

    """
    if file_path is not None:
        return load_hdf5_single_value_from_file(variable_name, file_path)

    if hdf_object is not None:
        return load_hdf5_single_value_from_object(variable_name, hdf_object)

    return ValueError("You must specify either a file_path or an hdf_object to read")


def load_hdf5_single_value_from_file(variable_name: str, file_path: str or AnyPath):
    """A function to read in the dark correction data from an HDF5 file.
    This will change as we store our data in NetCDF files

    Parameters
    ----------
    variable_name: str
        The name of the variable in the HDF5 file
    file_path: str or AnyPath
        The path to the HDF5 file

    """

    data_object = h5py.File(file_path, "r")
    dark_offset = load_hdf5_single_value_from_object(variable_name, data_object)
    data_object.close()

    return dark_offset


def load_hdf5_single_value_from_object(variable_name: str, hdf_dataset: h5py.Dataset):
    """A function to read in the dark correction data from an HDF5 file.
    This will change as we store our data in NetCDF files

    Parameters
    ----------
    variable_name: str
        The name of the variable in the HDF5 file
    hdf_object: h5py.Dataset, Optional
        The h5py Dataset to extract a value from

    Returns
    -------
    dark_offset: np.ndarray
        The dark offset data
    """
    return hdf_dataset[variable_name][()]
