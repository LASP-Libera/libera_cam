import pytest
import sys
from pathlib import Path


@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'