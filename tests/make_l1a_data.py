#! /usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

CCSDS_EPOCH = np.datetime64("1958-01-01T00:00:00")


def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent / "test_data"


def convert_to_multipart(
    time_data: npt.NDArray[np.datetime64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Convert a numpy array of datetime64 elements to a tuple containing
    the day, millisecond, and microsecond int64 parts of the CCSDS multipart time format.

    Parameters
    ----------
    time_data: npt.NDArray[np.datetime64]
        The input datetime64 array

    Returns
    -------
    tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]
        tuple of arrays containing the CCSDS multipart time
    """
    # compute day, ms, us variables
    delta_time = (time_data - CCSDS_EPOCH).astype(np.int64)
    one_day_ns = np.timedelta64(1, "D").astype("timedelta64[ns]").astype(np.int64)
    day_data, sec_data = np.divmod(delta_time, one_day_ns)
    # convert ns to ms with remainder in ns
    ms_data, ns_data = np.divmod(sec_data, 1_000_000)
    # convert remainder to us and round to int
    us_data = np.rint(ns_data / 1000).astype("int64")

    return day_data, ms_data, us_data


def make_l1a_data(input_path: Path, output_path: Path, copies: int = 2):
    """
    Stores a netcdf file created by concatenating n copies of each data variable
    array in an input netcdf file. The input dataset is assumed in the format of
    a LIBERA_L1A file.

    Then regenerates the time variables to form a fixed-cadence sequence covering
    the same time range as the original file. Thus, the cadence is
    a factor of approximately `copies` smaller than the original cadence.

    Parameters
    ----------
    input_path: Path
        Full path to the input netcdf file
    output_path: Path
        Full path to the output netcdf file
    copies: int
        Number of copies of each data variable
    """
    ds = xr.open_dataset(input_path)

    # Get the earliest and latest times in PACKET_ICIE_TIME (datetime64)
    icie_time_var = "PACKET_ICIE_TIME"
    min_icie_time = ds[icie_time_var].values[0].astype("int64")
    max_icie_time = ds[icie_time_var].values[-1].astype("int64")
    orig_len = len(ds[icie_time_var])

    if copies < 0:
        # use a slice of time based on abs(copies) as a percent
        max_icie_time = min_icie_time + int(abs(copies) / 100 * (max_icie_time - min_icie_time))
        min_dt = min_icie_time.astype("datetime64[ns]")
        max_dt = max_icie_time.astype("datetime64[ns]")
        mask = (ds[icie_time_var] >= min_dt) & (ds[icie_time_var] <= max_dt)
        ds_slice = ds.isel(packet=mask)
        ds_slice.to_netcdf(output_path)
        return

    # New cadence covers the same time range with copies times more timesteps
    new_len = orig_len * copies
    new_cadence = float(max_icie_time - min_icie_time) / new_len

    # Start tiled Dataset with a new contiguous time coordinate
    packet_dim = "packet"
    icie_day_var = "ICIE__TM_DAY_WFOV_SCI"
    icie_ms_var = "ICIE__TM_MS_WFOV_SCI"
    icie_us_var = "ICIE__TM_US_WFOV_SCI"
    all_time_vars = frozenset([packet_dim, icie_day_var, icie_ms_var, icie_us_var])

    # Build new time sequence as datetime64 objects
    time_data = np.array(min_icie_time + np.arange(new_len) * new_cadence).astype("datetime64[ns]")

    # compute day, ms, us variables
    day_data, ms_data, us_data = convert_to_multipart(time_data)

    # create new dataset with the time vars on packet dimension
    ds_tiled = xr.Dataset(
        data_vars={
            icie_time_var: (packet_dim, time_data),
            icie_day_var: (packet_dim, day_data),
            icie_ms_var: (packet_dim, ms_data),
            icie_us_var: (packet_dim, us_data),
        }
    )
    for var in ds.data_vars:
        if ds[var].dims:
            if var not in all_time_vars:
                ds_tiled[var] = xr.DataArray(np.tile(ds[var].data, copies), dims=(packet_dim,))

    # skip the metadata, not needed for performance testing

    ds_tiled.to_netcdf(output_path)


def main():
    ditl_data_path = test_data_path() / "DITL_short"
    output_folder = Path("/data")
    original_filename = Path("LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc")
    input_path = ditl_data_path / original_filename

    # negative copies takes that percentage of the input dataset time range
    copies = -25
    output_path = output_folder / f"{original_filename.stem}_{copies}.nc"
    make_l1a_data(input_path, output_path, copies=copies)


if __name__ == "__main__":
    main()
