import os
import sys
import glob
import datetime
import numpy as np

from .dark import get_dark_offset_syn
from .nlin import get_nlin_factor_syn
from .flfd import get_flfd_factor_syn
from .rad  import get_rad_factor_syn

__all__ = [
        'dn2rad',
        ]

def dn2rad(cnt_obs, int_time):

    # dark correction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dark_offset = get_dark_offset_syn(int_time)
    cnt_dark_corr = cnt_obs - dark_offset
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # non-linearity correction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    nlin_factor = get_nlin_factor_syn(cnt_dark_corr)
    cnt_nlin_corr = cnt_dark_corr * nlin_factor
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # flat-fielding correction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    flfd_factor = get_flfd_factor_syn(cnt_nlin_corr)
    cnt_flfd_corr = cnt_nlin_corr * flfd_factor
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # apply radiometric response
    #╭────────────────────────────────────────────────────────────────────────────╮#
    rad_factor = get_rad_factor_syn(int_time)
    rad_cam = cnt_flfd_corr * rad_factor
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return rad_cam


if __name__ == '__main__':

    pass
