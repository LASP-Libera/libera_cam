import numpy as np
import pytest

from libera_cam.correction_tools import get_flat_field_factor


@pytest.mark.parametrize("use_synthetic", [True, False])
def test_get_flat_field_factor(use_synthetic):
    if use_synthetic:
        synth_data = get_flat_field_factor(use_synthetic=True)
        np.testing.assert_equal(synth_data, 1)
    else:
        with pytest.raises(NotImplementedError):
            get_flat_field_factor(use_synthetic=False)
