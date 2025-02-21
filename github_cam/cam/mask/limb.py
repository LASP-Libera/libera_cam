import os
import sys
import glob
import datetime
import numpy as np

import cam.util
import cam.common

__all__ = [
        'mask_limb',
        ]

_data_ = cam.util.cal_vza_vaa(cam.common.FPA_NX, cam.common.FPA_NY, cam.common.FPA_DX, cam.common.FPA_DY, coef_dist2ang=cam.common.COEF_DIST2ANG)
_vza_limit_ = cam.common.VZA_LIMIT
_vza_range_ = [53.0, _vza_limit_]



def mask_limb(
        mask=None,
        vza_range=_vza_range_,
        vza_limit=_vza_limit_,
        ):

    if mask is None:
        mask = np.ones_like(_data_['vza'], dtype=np.int16)

    Nx, Ny = mask.shape

    # mask limb
    #╭────────────────────────────────────────────────────────────────────────────╮#
    mask[(_data_['vza']>=vza_range[0]) & (_data_['vza']<=vza_range[-1])] = 0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if vza_limit is not None:
        mask[_data_['vza']>vza_limit] = 1

    return mask



if __name__ == '__main__':

    mask = mask_limb()
    pass
