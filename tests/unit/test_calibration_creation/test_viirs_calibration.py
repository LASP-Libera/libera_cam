import pytest

from libera_cam.calibration_creation.viirs_calibrations import generate_viirs_calibration_factor


def test_generate_viirs_calibration_factor():
    with pytest.raises(NotImplementedError):
        generate_viirs_calibration_factor()
