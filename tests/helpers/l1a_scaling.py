"""Helpers to scale L1A NetCDF fixtures for integration and profiling tests."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from libera_utils.l1a.wfov_image_metadata import CAMERA_TIME_COORD
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
    """Scale or slice an image-centric L1A WFOV file along ``CAMERA_TIME``.

    Stores a NetCDF created by tiling each ``CAMERA_TIME``-indexed variable
    ``copies`` times, then regenerating ``CAMERA_TIME`` as a fixed-cadence
    sequence covering the original time range.

    Parameters
    ----------
    input_path: Path
        Full path to the input NetCDF file (image-enhanced L1A).
    output_path: Path
        Full path to the output NetCDF file.
    copies: int
        Number of copies of each image. Negative values select a percentage
        slice of the ``CAMERA_TIME`` range (e.g. ``-50`` keeps the first half).
    """
    ds = xr.open_dataset(input_path)

    if CAMERA_TIME_COORD not in ds.dims and CAMERA_TIME_COORD not in ds.coords:
        raise ValueError(
            f"L1A fixture must be image-centric with {CAMERA_TIME_COORD}; "
            "packet-only products are no longer supported by make_extended_l1a."
        )

    camera_times = ds[CAMERA_TIME_COORD].values
    min_time = camera_times[0].astype("int64")
    max_time = camera_times[-1].astype("int64")
    orig_len = len(camera_times)

    if copies < 0:
        max_time = min_time + int(abs(copies) / 100 * (max_time - min_time))
        min_dt = min_time.astype("datetime64[ns]")
        max_dt = max_time.astype("datetime64[ns]")
        mask = (ds[CAMERA_TIME_COORD] >= min_dt) & (ds[CAMERA_TIME_COORD] <= max_dt)
        ds.isel({CAMERA_TIME_COORD: mask}).to_netcdf(output_path)
        return

    new_len = orig_len * copies
    time_data = np.linspace(min_time, max_time, new_len, dtype=np.int64, endpoint=False).astype("datetime64[ns]")

    data_vars = {}
    for var in ds.data_vars:
        if CAMERA_TIME_COORD not in ds[var].dims:
            continue
        tiled = np.tile(ds[var].values, (copies,) + (1,) * (ds[var].ndim - 1))
        data_vars[var] = (ds[var].dims, tiled)

    coords = {CAMERA_TIME_COORD: (CAMERA_TIME_COORD, time_data)}
    for coord_name, coord in ds.coords.items():
        if coord_name == CAMERA_TIME_COORD:
            continue
        if CAMERA_TIME_COORD in coord.dims:
            tiled = np.tile(coord.values, (copies,) + (1,) * (coord.ndim - 1))
            coords[coord_name] = (coord.dims, tiled)
        else:
            coords[coord_name] = coord

    ds_tiled = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(ds.attrs))
    ds_tiled.to_netcdf(output_path)


def multipart_fields_from_dt64(
    time_data: npt.NDArray[np.datetime64],
) -> pd.DataFrame:
    """Build a DataFrame of CCSDS multipart fields suitable for ``multipart_to_dt64``."""
    day_data, ms_data, us_data = _dt64_to_day_ms_us(time_data)
    return pd.DataFrame({"day": day_data, "ms": ms_data, "us": us_data})
