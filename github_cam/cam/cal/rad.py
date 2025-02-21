import os
import sys
import glob
import datetime
import numpy as np


__all__ = [
        'get_rad_factor',
        'get_rad_factor_syn',
        ]


def get_rad_factor(
        int_time,
        band_width=20.0,
        coef=2.4e4,
        ):

    scale_factor = get_rad_factor_syn(int_time, band_width=band_width, coef=coef)

    return scale_factor

def get_rad_factor_syn(
        int_time,
        band_width=20.0,
        coef=2.4e4, # email exchange, 0.5 [radiance] approx. 800 DN at 2 ms
        ):

    scale_factor = 1.0 / (band_width * coef * (int_time/1000.0))

    return scale_factor


if __name__ == '__main__':

    pass
