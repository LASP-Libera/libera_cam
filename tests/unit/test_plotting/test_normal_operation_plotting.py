import datetime

import h5py
import numpy as np
import pytest
import xarray as xr

import libera_cam.utils.hdf5_io as hdf5_io
from libera_cam.camera import convert_dn_to_radiance
from libera_cam.constants import IntegrationTime
from libera_cam.plotting_tools.normal_operations import plot_observed_vs_true_plus_relative_difference


@pytest.mark.parametrize(
    ("integration_time", "variable_name"), [(IntegrationTime.LONG, "cnt_obs1"), (IntegrationTime.SHORT, "cnt_obs0")]
)
def test_plot_observed_vs_true_plus_relative_difference(
    test_data_path, local_data_path, integration_time, variable_name
):
    test_data = h5py.File(test_data_path / "camera_raw_count_example.h5", "r")
    pixel_counts = hdf5_io.load_hdf5_variable_from_object(variable_name, test_data)

    reference_data = h5py.File(test_data_path / "viirs_known_result.h5", "r")
    ref_radiance = hdf5_io.load_hdf5_single_value_from_object("rad", reference_data)

    obs_julian_day = hdf5_io.load_hdf5_single_value_from_object("_metadata_/jday0", reference_data)
    obs_datetime = datetime.datetime(1, 1, 1) + datetime.timedelta(
        seconds=np.round(((obs_julian_day - 1) * 86400.0), decimals=0)
    )

    # Wrap inputs in DataArray as expected by camera.py
    # pixel_counts is likely (2048, 2048), we add time dim
    dn_da = xr.DataArray(pixel_counts[np.newaxis, ...], dims=("camera_time", "y", "x"))
    # IntegrationTime is a scalar Enum
    int_time_da = xr.DataArray(np.full((1, 2048, 2048), integration_time.value), dims=("camera_time", "y", "x"))

    observed_radiance_da = convert_dn_to_radiance(dn_da, int_time_da, use_synthetic=True)
    observed_radiance = observed_radiance_da.values.squeeze()

    observed_radiance[np.isnan(ref_radiance)] = np.nan

    plot_observed_vs_true_plus_relative_difference(ref_radiance, observed_radiance, integration_time, obs_datetime)

    # Uncomment below to view the plots during non-automated testing
    # plt.show()

    test_data.close()
    reference_data.close()
