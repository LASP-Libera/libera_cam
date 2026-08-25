"""Convert WFOV FPGA exposure register values to milliseconds."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from libera_cam.constants import WFOV_DEFAULT_CLK_PER_VALUE

# Constants from the L1A product definition for WFOV_FPGA_ACTUAL_EXP_TIME_*.
_ACTUAL_EXP_OFFSET = 0.43 * 20
_ACTUAL_EXP_LINE_FACTOR = 129.0


def actual_exposure_counts_to_ms(
    value: npt.ArrayLike,
) -> npt.NDArray[np.floating] | float:
    """Convert FPGA actual exposure register counts to milliseconds.

    Uses the L1A-documented equation::

        ms = (value + 0.43 * 20) * 129.0 * 0.15625 / 1000

    Parameters
    ----------
    value :
        Raw FPGA actual exposure register count(s).

    Returns
    -------
    float or ndarray
        Exposure duration in milliseconds.
    """
    arr = np.asarray(value, dtype=np.float64)
    result = (arr + _ACTUAL_EXP_OFFSET) * _ACTUAL_EXP_LINE_FACTOR * WFOV_DEFAULT_CLK_PER_VALUE / 1000.0
    if result.ndim == 0:
        return float(result)
    return result.astype(np.float32)


def delta_exposure_counts_to_ms(
    value: npt.ArrayLike,
) -> npt.NDArray[np.floating] | float:
    """Convert FPGA DELTA_EXP register counts to milliseconds.

    Uses the clock-period conversion::

        us = value * WFOV_DEFAULT_CLK_PER_VALUE
        ms = us / 1000

    Parameters
    ----------
    value :
        Raw FPGA DELTA_EXP register count(s).

    Returns
    -------
    float or ndarray
        Delta duration in milliseconds.
    """
    arr = np.asarray(value, dtype=np.float64)
    result = arr * WFOV_DEFAULT_CLK_PER_VALUE / 1000.0
    if result.ndim == 0:
        return float(result)
    return result.astype(np.float32)
