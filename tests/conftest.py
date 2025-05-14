import sys
from pathlib import Path

import pytest


@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'


@pytest.fixture
def local_data_path():
    """Returns the Path to the calibration_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent.parent / 'libera_cam' / 'ground_calibration_data'
