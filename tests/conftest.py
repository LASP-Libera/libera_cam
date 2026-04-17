import sys
from pathlib import Path

import pytest


@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent / "test_data"


@pytest.fixture
def test_ditl_l1a_file_path(test_data_path):
    """Returns the Path to a sample L1A NetCDF file from the Day in the Life (DITL) test data"""
    return (
        test_data_path
        / "DITL_short"
        / "LIBERA_L1A_WFOV-SCI-DECODED_V5-4-2_20280215T135304_20280215T142141_R26021133743.nc"
    )


@pytest.fixture
def local_data_path():
    """Returns the Path to the calibration_data directory"""
    return Path(sys.modules[__name__.split(".")[0]].__file__).parent.parent / "libera_cam" / "ground_calibration_data"
