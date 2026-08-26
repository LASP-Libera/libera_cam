"""Helpers to scale L1A NetCDF fixtures for integration and profiling tests."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from libera_utils.time import CCSDS_EPOCH

CCSDS_EPOCH_NS = np.datetime64(CCSDS_EPOCH)


def _dt64_to_day_ms_us(
    time_data: npt.NDArray[np.datetime64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Derive CCSDS day/ms/us fields from datetime64 (inverse of ``multipart_to_dt64``)."""
    delta_ns = (time_data - CCSDS_EPOCH_NS).astype(np.int64)
    one_day_ns = np.timedelta64(1, "D").astype("timedelta64[ns]").astype(np.int64)
    day_data, within_day_ns = np.divmod(delta_ns, one_day_ns)
    ms_data, within_ms_ns = np.divmod(within_day_ns, 1_000_000)
    us_data = (within_ms_ns // 1000).astype(np.int64)
    return day_data, ms_data, us_data


def make_extended_l1a(input_path: Path, output_path: Path, copies: int = 2) -> None:
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

    icie_time_var = "PACKET_ICIE_TIME"
    packet_dim = ds[icie_time_var].dims[0]
    min_icie_time = ds[icie_time_var].values[0].astype("int64")
    max_icie_time = ds[icie_time_var].values[-1].astype("int64")
    orig_len = len(ds[icie_time_var])

    if copies < 0:
        max_icie_time = min_icie_time + int(abs(copies) / 100 * (max_icie_time - min_icie_time))
        min_dt = min_icie_time.astype("datetime64[ns]")
        max_dt = max_icie_time.astype("datetime64[ns]")
        mask = (ds[icie_time_var] >= min_dt) & (ds[icie_time_var] <= max_dt)
        ds_slice = ds.isel({packet_dim: mask})
        ds_slice.to_netcdf(output_path)
        return

    new_len = orig_len * copies

    icie_day_var = "ICIE__TM_DAY_WFOV_SCI"
    icie_ms_var = "ICIE__TM_MS_WFOV_SCI"
    icie_us_var = "ICIE__TM_US_WFOV_SCI"
    all_time_vars = frozenset([packet_dim, icie_time_var, icie_day_var, icie_ms_var, icie_us_var])

    time_data = np.linspace(min_icie_time, max_icie_time, new_len, dtype=np.int64, endpoint=False).astype(
        "datetime64[ns]"
    )
    day_data, ms_data, us_data = _dt64_to_day_ms_us(time_data)

    ds_tiled = xr.Dataset(
        data_vars={
            icie_time_var: (packet_dim, time_data),
            icie_day_var: (packet_dim, day_data),
            icie_ms_var: (packet_dim, ms_data),
            icie_us_var: (packet_dim, us_data),
        }
    )
    for var in ds.data_vars:
        if ds[var].dims and var not in all_time_vars:
            ds_tiled[var] = xr.DataArray(np.tile(ds[var].data, copies), dims=(packet_dim,))

    ds_tiled.to_netcdf(output_path)


def multipart_fields_from_dt64(
    time_data: npt.NDArray[np.datetime64],
) -> pd.DataFrame:
    """Build a DataFrame of CCSDS multipart fields suitable for ``multipart_to_dt64``."""
    day_data, ms_data, us_data = _dt64_to_day_ms_us(time_data)
    return pd.DataFrame({"day": day_data, "ms": ms_data, "us": us_data})
