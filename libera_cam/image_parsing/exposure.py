"""Convert WFOV FPGA exposure register values to milliseconds."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from libera_cam.constants import WFOV_DEFAULT_CLK_PER_VALUE

# Constants from the L1A product definition for WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_*.
_ACTUAL_EXP_OFFSET = 0.43 * 20
_ACTUAL_EXP_LINE_FACTOR = 129.0


def actual_exposure_counts_to_ms(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Convert FPGA actual exposure register counts to milliseconds.

    Uses the L1A-documented equation::

        ms = (value + 0.43 * 20) * 129.0 * 0.15625 / 1000

    Parameters
    ----------
    value :
        Raw FPGA actual exposure register count(s).

    Returns
    -------
    ndarray
        Exposure duration in milliseconds.
    """
    arr = np.asarray(value, dtype=np.float64)
    result = (arr + _ACTUAL_EXP_OFFSET) * _ACTUAL_EXP_LINE_FACTOR * WFOV_DEFAULT_CLK_PER_VALUE / 1000.0
    return result.astype(np.float32)


# TODO[LIBSDC-844]: The L1A product definition does not yet document this conversion the way it does for
# ``WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_*``. It is corroborated by the DITL fixture, where a raw
# 2240000 converts to 350.0 ms — the upper bound of the 111-350 ms dual-exposure lag documented
# for ``WFOV_FSW_HEADER_IMG_MODE``.
def delta_exposure_counts_to_ms(value: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Convert FPGA DELTA_EXP register counts to milliseconds.

    The FPGA DELTA_EXP register counts clock periods directly, with none of the offset or line
    factor the actual-exposure registers carry::

        ms = value * WFOV_DEFAULT_CLK_PER_VALUE / 1000

    Parameters
    ----------
    value :
        Raw FPGA DELTA_EXP register count(s).

    Returns
    -------
    ndarray
        Delta duration in milliseconds.
    """
    arr = np.asarray(value, dtype=np.float64)
    result = arr * WFOV_DEFAULT_CLK_PER_VALUE / 1000.0
    return result.astype(np.float32)
