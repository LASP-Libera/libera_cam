import os
import sys
import glob
import datetime
import numpy as np


import cam.common


__all__ = [
        'get_flfd_factor',
        'get_flfd_factor_syn',
        ]


def get_flfd_factor(
        ):

    scale_factor = get_flfd_factor_syn()

    return scale_factor

def get_flfd_factor_syn(
        Nbit=cam.common.NBIT,
        Nx=cam.common.FPA_NX,
        Ny=cam.common.FPA_NY,
        ):

    scale_factor = np.ones(
            (Nx, Ny),
            dtype=np.float32)

    return scale_factor


if __name__ == '__main__':

    pass
