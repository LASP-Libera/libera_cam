import os
import numpy as np

__all__ = ['FDIR_DATA', 'PLANET_RADIUS', \
           'FPA_NX', 'FPA_NY', 'FPA_DX', 'FPA_DY', \
           'COEF_ANG2DIST', 'COEF_DIST2ANG', \
           'VZA_LIMIT', \
           'NBIT',
           'DATA_COEF',
          ]

FDIR_DATA = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

PLANET_RADIUS = 6370997.0
FPA_NX = 2048
FPA_NY = 2048
FPA_DX = 0.0055
FPA_DY = 0.0055
COEF_DIST2ANG = np.array([1.397428e-02, -1.500364e-01, 6.352646e-01, -9.551312e-01,    9.042359, 0])
COEF_ANG2DIST = np.array([ 1.21367e-10,   2.60014e-09, -7.50181e-06,  -2.84788e-06, 1.17022e-01, 0 ])
VZA_LIMIT = 63.0

NBIT = 12

DATA_COEF = {}
