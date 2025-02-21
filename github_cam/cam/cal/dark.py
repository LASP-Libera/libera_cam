import os
import sys
import glob
import datetime
import numpy as np


import cam.common


__all__ = [
        'get_dark_offset',
        'get_dark_offset_syn',
        ]


def get_dark_offset(
        int_time,
        ):

    add_offset = get_dark_offset_syn(int_time)

    return add_offset

def get_dark_offset_syn(
        int_time,
        Nx=cam.common.FPA_NX,
        Ny=cam.common.FPA_NY,
        Nbit=cam.common.NBIT,
        ):

    int_time = int(int_time)

    dark_mean = {
            1: 100.0/2.0**Nbit,
            20: 200.0/2.0**Nbit,
            }

    dark_std = {
            1: 50.0/2.0**Nbit,
            20: 100.0/2.0**Nbit,
            }

    add_offset = 2.0**Nbit * np.random.normal(loc=dark_mean[int_time], scale=dark_std[int_time], size=(Nx, Ny))

    return add_offset


if __name__ == '__main__':

    pass
