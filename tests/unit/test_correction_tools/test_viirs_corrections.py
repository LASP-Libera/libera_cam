import pytest

from libera_cam.correction_tools import get_viirs_adjustment_factor


def test_get_viirs_adjustment_factor():
    with pytest.raises(NotImplementedError):
        get_viirs_adjustment_factor()
